from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from openai import OpenAI
import os
import json


class LoLRAG:
    def __init__(self, openai_api_key, pinecone_api_key, pinecone_index_name="lol-rag"):
        """Initialize all connections"""
        
        self.graph = Neo4jGraph(
            url="neo4j+s://f5886a71.databases.neo4j.io",
            username="neo4j",
            password="fFCqS5-Ak3ZLppY0p1e2rrL1ZN_FJwt7oH_McxlCh5o"
        )
        
        pc = Pinecone(api_key=pinecone_api_key)
        self.pinecone_index = pc.Index(pinecone_index_name)
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=openai_api_key)
        
        self._setup_neo4j_chain()
    
    def _setup_neo4j_chain(self):
        
        CYPHER_TEMPLATE = """Generate Cypher to query a graph database.

CRITICAL: All property values use Title Case:
- Roles: 'Mage', 'Tank', 'Fighter', 'Assassin', 'Support', 'Marksman'
- Crowd Control: 'Stun', 'Root', 'Slow', 'Grounded', 'Knockup', etc.
- Damage: 'Magic', 'Physical', 'True'
- Attack: 'Melee', 'Ranged'

Schema: {schema}
Question: {question}
Cypher:"""
        
        CYPHER_PROMPT = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_TEMPLATE
        )
        
        self.neo4j_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            verbose=False,  
            allow_dangerous_requests=True,
            return_intermediate_steps=True,
            cypher_prompt=CYPHER_PROMPT
        )
    
    def create_embedding(self, text):
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def pinecone_search(self, query, top_k=10, filter_type=None):
        """Semantic search in Pinecone"""
        query_embedding = self.create_embedding(query)
        query_filter = {'type': filter_type} if filter_type else None
        
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=query_filter
        )
        return results['matches']
    
    def analyze_query(self, question):
        """Decide retrieval strategy"""
        
        analysis_prompt = f"""
Analyze this League of Legends question: "{question}"

Decide which database to use:
- **neo4j**: Structured filters (roles, damage types, CC types, stats)
- **pinecone**: Semantic/conceptual ("fire champions", "burst damage", themes)
- **hybrid**: Both ("ranged mages with fire abilities")

Respond in JSON:
{{
    "strategy": "neo4j" | "pinecone" | "hybrid",
    "reasoning": "why",
    "semantic_query": "reformulated query" (if pinecone/hybrid),
    "filters": {{"role": "Mage"}} (if hybrid)
}}
"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a query analyzer. Return only valid JSON."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"strategy": "neo4j", "reasoning": "fallback"}
    
    def retrieve_context(self, question):
        """Master retrieval function"""
        
        analysis = self.analyze_query(question)
        strategy = analysis['strategy']
        
        context_parts = []
        sources = []
        metadata = {
            'strategy': strategy,
            'reasoning': analysis['reasoning']
        }
        
        if strategy == 'neo4j':
            result = self.neo4j_chain.invoke({"query": question})
            intermediate = result.get('intermediate_steps', [])
            
            if len(intermediate) >= 2:
                cypher_query = intermediate[0].get('query', 'N/A')
                raw_results = intermediate[1].get('context', [])
                
                metadata['cypher_query'] = cypher_query
                metadata['result_count'] = len(raw_results)
                
                if raw_results:
                    context_parts.append("**Neo4j Query Results:**")
                    context_parts.append(f"Cypher: {cypher_query}\n")
                    
                    for i, row in enumerate(raw_results[:50], 1):
                        formatted = ", ".join([f"{k}: {v}" for k, v in row.items()])
                        context_parts.append(f"{i}. {formatted}")
                    
                    sources.append(f"Neo4j: {len(raw_results)} results")
                else:
                    context_parts.append(f"No results found.\nQuery: {cypher_query}")
                    sources.append("Neo4j: 0 results")
            else:
                context_parts.append(f"Neo4j: {result.get('result', 'N/A')}")
                sources.append("Neo4j")
        
        elif strategy == 'pinecone':
            semantic_query = analysis.get('semantic_query', question)
            matches = self.pinecone_search(semantic_query, top_k=10)
            
            metadata['semantic_query'] = semantic_query
            metadata['result_count'] = len(matches)
            
            context_parts.append("**Semantic Search Results:**")
            for i, match in enumerate(matches[:5], 1):
                meta = match['metadata']
                if meta['type'] == 'ability':
                    context_parts.append(
                        f"{i}. {meta['champion_name']} - {meta['ability_name']} [{meta['slot']}]\n"
                        f"   {meta['text'][:150]}..."
                    )
                else:
                    context_parts.append(f"{i}. {meta['champion_name']} - {meta['title']}")
                sources.append(meta['champion_name'])
        
        elif strategy == 'hybrid':
            semantic_query = analysis.get('semantic_query', question)
            matches = self.pinecone_search(semantic_query, top_k=15)
            
            champion_ids = list(set([m['metadata']['champion_id'] for m in matches]))
            
            filters = analysis.get('filters', {})
            if filters:
                cypher_parts = ["MATCH (c:Champion) WHERE c.id IN $champion_ids"]
                params = {'champion_ids': champion_ids}
                
                if 'role' in filters:
                    cypher_parts.append("MATCH (c)-[:HAS_ROLE]->(r:Role {name: $role})")
                    params['role'] = filters['role']
                
                if 'attack_type' in filters:
                    cypher_parts.append("MATCH (c)-[:ATTACKS_WITH]->(at:AttackType {name: $attack_type})")
                    params['attack_type'] = filters['attack_type']
                
                cypher_parts.append("RETURN DISTINCT c.id as id")
                cypher_query = " ".join(cypher_parts)
                
                filtered = self.graph.query(cypher_query, params=params)
                filtered_ids = {c['id'] for c in filtered}
                matches = [m for m in matches if m['metadata']['champion_id'] in filtered_ids]
            
            metadata['semantic_query'] = semantic_query
            metadata['filters'] = filters
            metadata['result_count'] = len(matches)
            
            context_parts.append("**Hybrid Search Results:**")
            for i, match in enumerate(matches[:5], 1):
                meta = match['metadata']
                if meta['type'] == 'ability':
                    context_parts.append(
                        f"{i}. {meta['champion_name']} - {meta['ability_name']} [{meta['slot']}]\n"
                        f"   Score: {match['score']:.3f}\n"
                        f"   {meta['text'][:150]}..."
                    )
                sources.append(meta['champion_name'])
        
        context = "\n\n".join(context_parts)
        
        return {
            'context': context,
            'sources': sources[:5],
            'metadata': metadata
        }
    
    def generate_answer(self, question, context_data):
        """Generate answer using GPT-4"""
        
        system_prompt = """
You are a League of Legends expert assistant.

Answer questions accurately using the provided context.
Be concise but informative.
Cite specific champions/abilities when relevant.
If context is insufficient, say so honestly.
"""
        
        user_prompt = f"""
Question: {question}

Context:
{context_data['context']}

Answer:
"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def ask_question(self, question):
        """
        Main Q&A function
        
        Returns:
            dict with:
            - question: Original question
            - answer: Generated answer
            - metadata: Strategy, sources, etc.
        """
        context_data = self.retrieve_context(question)
        answer = self.generate_answer(question, context_data)
        
        return {
            'question': question,
            'answer': answer,
            'strategy': context_data['metadata']['strategy'],
            'reasoning': context_data['metadata']['reasoning'],
            'sources': context_data['sources'],
            'metadata': context_data['metadata']
        }
