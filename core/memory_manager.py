"""
Memory management module for storing and retrieving conversation history and analysis data
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

# Optional: Vector embeddings for semantic search
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

class MemoryManager:
    """Handles short-term and long-term memory for the application."""
    
    def __init__(self, db_path: str = "data/crypto_memory.db"):
        self.db_path = db_path
        self.embedding_model = None
        self.faiss_index = None
        self.vector_metadata = []
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Initialize embeddings if available
        if EMBEDDINGS_AVAILABLE:
            self._initialize_embeddings()
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Analysis cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    result_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    session_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT NOT NULL,
                    predicted_price REAL,
                    actual_price REAL,
                    prediction_date DATETIME,
                    evaluation_date DATETIME,
                    model_type TEXT,
                    accuracy_score REAL
                )
            """)
            
            conn.commit()
    
    def _initialize_embeddings(self):
        """Initialize sentence transformer model for semantic search."""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize FAISS index
            embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            
            # Load existing embeddings if available
            self._load_existing_embeddings()
            
        except Exception as e:
            print(f"Failed to initialize embeddings: {e}")
            self.embedding_model = None
            self.faiss_index = None
    
    def save_interaction(self, session_id: str, query: str, 
                        result: Dict, role: str = "user"):
        """Save user interaction to memory."""
        metadata = {
            "query_type": result.get("query_type", "analysis"),
            "coin_id": result.get("market", {}).get("coin", "unknown"),
            "success": result.get("success", False)
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (session_id, role, message, metadata)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, query, json.dumps(metadata)))
            
            # Also save assistant response
            if result.get("success"):
                assistant_message = f"Analysis completed for {metadata['coin_id']}"
                cursor.execute("""
                    INSERT INTO conversations (session_id, role, message, metadata)
                    VALUES (?, ?, ?, ?)
                """, (session_id, "assistant", assistant_message, json.dumps(result)))
            
            conn.commit()
        
        # Add to vector index if embeddings available
        if self.embedding_model:
            self._add_to_vector_index(query, result)
    
    def get_conversation_history(self, session_id: str, 
                                limit: int = 10) -> List[Dict]:
        """Retrieve conversation history for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, message, metadata, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "role": row[0],
                    "message": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "timestamp": row[3]
                })
            
            return list(reversed(history))  # Return in chronological order
    
    def cache_analysis_result(self, coin_id: str, query_hash: str,
                             result: Dict, ttl_hours: int = 1):
        """Cache analysis result for faster retrieval."""
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO analysis_cache 
                (coin_id, query_hash, result_data, expires_at)
                VALUES (?, ?, ?, ?)
            """, (coin_id, query_hash, json.dumps(result), expires_at))
            conn.commit()
    
    def get_cached_analysis(self, coin_id: str, query_hash: str) -> Optional[Dict]:
        """Retrieve cached analysis result if not expired."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT result_data FROM analysis_cache
                WHERE coin_id = ? AND query_hash = ? AND expires_at > ?
            """, (coin_id, query_hash, datetime.now()))
            
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None
    
    def save_user_preferences(self, session_id: str, preferences: Dict):
        """Save user preferences."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_preferences (session_id, preferences)
                VALUES (?, ?)
            """, (session_id, json.dumps(preferences)))
            conn.commit()
    
    def get_user_preferences(self, session_id: str) -> Dict:
        """Retrieve user preferences."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT preferences FROM user_preferences WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            return json.loads(row[0]) if row else {}
    
    def find_similar_queries(self, query: str, limit: int = 5) -> List[Dict]:
        """Find similar queries using semantic search."""
        if not self.embedding_model or not self.faiss_index:
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([query])
            
            # Search for similar embeddings
            distances, indices = self.faiss_index.search(query_embedding, limit)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.vector_metadata):
                    metadata = self.vector_metadata[idx]
                    results.append({
                        "query": metadata["query"],
                        "result": metadata["result"],
                        "similarity": 1.0 / (1.0 + distance),  # Convert to similarity score
                        "timestamp": metadata["timestamp"]
                    })
            
            return results
            
        except Exception as e:
            print(f"Semantic search failed: {e}")
            return []
    
    def _add_to_vector_index(self, query: str, result: Dict):
        """Add query and result to vector index."""
        if not self.embedding_model:
            return
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([query])
            
            # Add to FAISS index
            self.faiss_index.add(embedding.astype('float32'))
            
            # Store metadata
            self.vector_metadata.append({
                "query": query,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save to disk periodically
            if len(self.vector_metadata) % 10 == 0:
                self._save_embeddings()
                
        except Exception as e:
            print(f"Failed to add to vector index: {e}")
    
    def _save_embeddings(self):
        """Save FAISS index and metadata to disk."""
        try:
            faiss_path = "data/faiss_index.bin"
            metadata_path = "data/vector_metadata.json"
            
            faiss.write_index(self.faiss_index, faiss_path)
            
            with open(metadata_path, 'w') as f:
                json.dump(self.vector_metadata, f)
                
        except Exception as e:
            print(f"Failed to save embeddings: {e}")
    
    def _load_existing_embeddings(self):
        """Load existing FAISS index and metadata from disk."""
        try:
            faiss_path = "data/faiss_index.bin"
            metadata_path = "data/vector_metadata.json"
            
            if os.path.exists(faiss_path) and os.path.exists(metadata_path):
                self.faiss_index = faiss.read_index(faiss_path)
                
                with open(metadata_path, 'r') as f:
                    self.vector_metadata = json.load(f)
                    
                print(f"Loaded {len(self.vector_metadata)} embeddings from disk")
                
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
    
    def get_analysis_statistics(self) -> Dict:
        """Get statistics about stored analyses."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total conversations
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]
            
            # Unique sessions
            cursor.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
            unique_sessions = cursor.fetchone()[0]
            
            # Most analyzed coins
            cursor.execute("""
                SELECT 
                    JSON_EXTRACT(metadata, '$.coin_id') as coin_id,
                    COUNT(*) as count
                FROM conversations 
                WHERE JSON_EXTRACT(metadata, '$.coin_id') IS NOT NULL
                GROUP BY coin_id
                ORDER BY count DESC
                LIMIT 5
            """)
            popular_coins = cursor.fetchall()
            
            # Cache statistics
            cursor.execute("SELECT COUNT(*) FROM analysis_cache WHERE expires_at > ?", 
                          (datetime.now(),))
            active_cache_entries = cursor.fetchone()[0]
            
            return {
                "total_conversations": total_conversations,
                "unique_sessions": unique_sessions,
                "popular_coins": popular_coins,
                "active_cache_entries": active_cache_entries,
                "vector_embeddings": len(self.vector_metadata) if self.vector_metadata else 0
            }
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old conversation data and expired cache entries."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Remove old conversations
            cursor.execute("""
                DELETE FROM conversations WHERE timestamp < ?
            """, (cutoff_date,))
            
            # Remove expired cache entries
            cursor.execute("""
                DELETE FROM analysis_cache WHERE expires_at < ?
            """, (datetime.now(),))
            
            conn.commit()
            
            print(f"Cleaned up data older than {days_to_keep} days")