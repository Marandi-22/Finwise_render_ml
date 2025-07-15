import logging
import json
import threading
from typing import Dict, Any, Optional
from pathlib import Path
import time

class CacheManager:
    """Thread-safe cache manager for JSON data"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._last_modified: Dict[str, float] = {}
        
    def get_data(self, file_path: Path, default: Any = None) -> Any:
        """Get cached data or load from file"""
        file_key = str(file_path)
        
        with self._lock:
            try:
                # Check if file exists
                if not file_path.exists():
                    logging.warning(f"File not found: {file_path}")
                    return default or {}
                
                # Check if cache is still valid
                current_mtime = file_path.stat().st_mtime
                if (file_key in self._cache and 
                    file_key in self._last_modified and
                    self._last_modified[file_key] == current_mtime):
                    return self._cache[file_key]
                
                # Load fresh data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Update cache
                self._cache[file_key] = data
                self._last_modified[file_key] = current_mtime
                
                logging.info(f"Loaded data from {file_path}")
                return data
                
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON in {file_path}: {e}")
                return default or {}
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                return default or {}

# Global cache instance
cache_manager = CacheManager()