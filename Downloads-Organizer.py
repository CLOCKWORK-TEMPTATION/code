# pip install google-genai tqdm pillow watchdog
import os
import shutil
import json
import time
import hashlib
import logging
import threading
import queue
import sqlite3
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import piexif
from google import genai
from google.genai import types

# ==========================================
# Enhanced Configuration & Setup
# ==========================================
@dataclass
class Config:
    API_KEY: str = "YOUR_GEMINI_API_KEY"
    TARGET_FOLDER: str = r"C:\Users\moham\Downloads"
    LOG_FILE: str = os.path.join(TARGET_FOLDER, ".organizer_log.json")
    CACHE_DB: str = os.path.join(TARGET_FOLDER, ".organizer_cache.db")
    CONFIG_FILE: str = os.path.join(TARGET_FOLDER, ".organizer_config.json")
    BATCH_SIZE: int = 30
    MAX_RETRIES: int = 3
    DRY_RUN: bool = False
    MAX_WORKERS: int = 4  # Number of parallel workers
    AUTO_ORGANIZE: bool = False  # Enable automatic organization
    ORGANIZE_INTERVAL: str = "daily"  # When to auto-organize: hourly, daily, weekly
    COMPRESS_OLD_FILES: bool = False  # Compress files older than X days
    COMPRESS_DAYS: int = 30  # Days after which to compress files
    ENABLE_WATCHER: bool = False  # Enable real-time file watching
    SHOW_PREVIEW: bool = True  # Show preview before organizing
    
    # File types to read content from for AI categorization
    READABLE_EXTS: set = frozenset({'.txt', '.py', '.js', '.md', '.json', '.csv', '.html', '.log', '.xml', '.yaml', '.yml'})
    
    # File types to extract metadata from
    METADATA_EXTS: set = frozenset({'.jpg', '.jpeg', '.png', '.tiff', '.pdf', '.mp3', '.mp4', '.docx', '.xlsx'})
    
    # Custom categorization rules
    CUSTOM_RULES: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.CUSTOM_RULES is None:
            self.CUSTOM_RULES = {
                'Images': ['.jpg', '.jpeg', '.png', '.webp', '.svg', '.gif', '.bmp', '.tiff'],
                'Videos': ['.mp4', '.mkv', '.mov', '.avi', '.wmv', '.flv', '.webm'],
                'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
                'Documents': ['.pdf', '.docx', '.txt', '.rtf', '.odt'],
                'Spreadsheets': ['.xlsx', '.xls', '.csv', '.ods'],
                'Presentations': ['.pptx', '.ppt', '.odp'],
                'Executables': ['.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm'],
                'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go'],
                'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'],
                'Fonts': ['.ttf', '.otf', '.woff', '.woff2'],
                'Ebooks': ['.epub', '.mobi', '.azw', '.azw3']
            }
    
    @classmethod
    def load(cls, config_file: str = None) -> 'Config':
        """Load configuration from file"""
        if config_file is None:
            config_file = cls.CONFIG_FILE
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                logging.error(f"Failed to load config: {e}")
        
        return cls()
    
    def save(self, config_file: str = None):
        """Save configuration to file"""
        if config_file is None:
            config_file = self.CONFIG_FILE
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")

# Enhanced logging setup
def setup_logging(log_level=logging.INFO):
    """Setup logging with both file and console handlers"""
    log_dir = os.path.join(Config.TARGET_FOLDER, ".organizer_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"organizer_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("SmartOrganizer")

logger = setup_logging()

# ==========================================
# Database Cache for AI Responses
# ==========================================
class ResponseCache:
    """SQLite cache for AI responses to improve performance"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_cache (
                    file_hash TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_ext TEXT,
                    file_size INTEGER,
                    snippet TEXT,
                    category TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_responses (
                    request_hash TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def get_file_category(self, file_info: Dict) -> Optional[str]:
        """Get cached category for a file"""
        file_hash = self._get_file_hash(file_info)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT category FROM file_cache WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                return result[0]
        
        return None
    
    def cache_file_category(self, file_info: Dict, category: str):
        """Cache a file's category"""
        file_hash = self._get_file_hash(file_info)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_cache 
                (file_hash, file_name, file_ext, file_size, snippet, category)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_hash,
                    file_info['name'],
                    file_info['ext'],
                    file_info['size_kb'],
                    file_info.get('snippet', ''),
                    category
                )
            )
    
    def get_ai_response(self, request_hash: str) -> Optional[str]:
        """Get cached AI response"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response FROM ai_responses WHERE request_hash = ?",
                (request_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                return result[0]
        
        return None
    
    def cache_ai_response(self, request_hash: str, response: str):
        """Cache an AI response"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ai_responses (request_hash, response) VALUES (?, ?)",
                (request_hash, response)
            )
    
    def _get_file_hash(self, file_info: Dict) -> str:
        """Generate a hash for file identification"""
        content = f"{file_info['name']}|{file_info['size_kb']}|{file_info.get('snippet', '')[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def cleanup_old_entries(self, days: int = 30):
        """Remove cache entries older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM file_cache WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            
            conn.execute(
                "DELETE FROM ai_responses WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )

# ==========================================
# File Metadata Extractor
# ==========================================
class MetadataExtractor:
    """Extract metadata from various file types"""
    
    @staticmethod
    def extract_metadata(file_path: Path) -> Dict:
        """Extract metadata from a file"""
        ext = file_path.suffix.lower()
        
        if ext in {'.jpg', '.jpeg', '.tiff'}:
            return MetadataExtractor._extract_image_metadata(file_path)
        elif ext == '.png':
            return MetadataExtractor._extract_png_metadata(file_path)
        elif ext == '.pdf':
            return MetadataExtractor._extract_pdf_metadata(file_path)
        elif ext in {'.mp3', '.flac'}:
            return MetadataExtractor._extract_audio_metadata(file_path)
        elif ext in {'.mp4', '.avi', '.mkv', '.mov'}:
            return MetadataExtractor._extract_video_metadata(file_path)
        elif ext in {'.docx', '.xlsx', '.pptx'}:
            return MetadataExtractor._extract_office_metadata(file_path)
        
        return {}
    
    @staticmethod
    def _extract_image_metadata(file_path: Path) -> Dict:
        """Extract metadata from image files"""
        try:
            if file_path.suffix.lower() in {'.jpg', '.jpeg', '.tiff'}:
                exif_dict = piexif.load(str(file_path))
                
                metadata = {
                    'type': 'image',
                    'camera': None,
                    'date_taken': None,
                    'location': None,
                    'dimensions': None
                }
                
                # Camera info
                if '0th' in exif_dict and piexif.ImageIFD.Make in exif_dict['0th']:
                    metadata['camera'] = f"{exif_dict['0th'][piexif.ImageIFD.Make].decode()} {exif_dict['0th'].get(piexif.ImageIFD.Model, b'').decode()}"
                
                # Date taken
                if 'Exif' in exif_dict and piexif.ExifIFD.DateTimeOriginal in exif_dict['Exif']:
                    metadata['date_taken'] = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode()
                
                # GPS info
                if 'GPS' in exif_dict:
                    lat = MetadataExtractor._get_gps_coords(exif_dict['GPS'], piexif.GPSIFD.GPSLatitude)
                    lon = MetadataExtractor._get_gps_coords(exif_dict['GPS'], piexif.GPSIFD.GPSLongitude)
                    if lat and lon:
                        metadata['location'] = f"{lat}, {lon}"
                
                # Dimensions
                with Image.open(file_path) as img:
                    metadata['dimensions'] = f"{img.width}x{img.height}"
                
                return metadata
        except Exception as e:
            logger.debug(f"Failed to extract image metadata: {e}")
        
        return {'type': 'image'}
    
    @staticmethod
    def _extract_png_metadata(file_path: Path) -> Dict:
        """Extract metadata from PNG files"""
        try:
            with Image.open(file_path) as img:
                return {
                    'type': 'image',
                    'dimensions': f"{img.width}x{img.height}"
                }
        except Exception as e:
            logger.debug(f"Failed to extract PNG metadata: {e}")
        
        return {'type': 'image'}
    
    @staticmethod
    def _extract_pdf_metadata(file_path: Path) -> Dict:
        """Extract metadata from PDF files"""
        try:
            # This is a placeholder - in a real implementation, you'd use PyPDF2 or similar
            return {
                'type': 'document',
                'format': 'pdf'
            }
        except Exception as e:
            logger.debug(f"Failed to extract PDF metadata: {e}")
        
        return {'type': 'document', 'format': 'pdf'}
    
    @staticmethod
    def _extract_audio_metadata(file_path: Path) -> Dict:
        """Extract metadata from audio files"""
        try:
            # This is a placeholder - in a real implementation, you'd use mutagen or similar
            return {
                'type': 'audio',
                'format': file_path.suffix.lower()[1:]
            }
        except Exception as e:
            logger.debug(f"Failed to extract audio metadata: {e}")
        
        return {'type': 'audio', 'format': file_path.suffix.lower()[1:]}
    
    @staticmethod
    def _extract_video_metadata(file_path: Path) -> Dict:
        """Extract metadata from video files"""
        try:
            # This is a placeholder - in a real implementation, you'd use ffmpeg-python or similar
            return {
                'type': 'video',
                'format': file_path.suffix.lower()[1:]
            }
        except Exception as e:
            logger.debug(f"Failed to extract video metadata: {e}")
        
        return {'type': 'video', 'format': file_path.suffix.lower()[1:]}
    
    @staticmethod
    def _extract_office_metadata(file_path: Path) -> Dict:
        """Extract metadata from Office documents"""
        try:
            # This is a placeholder - in a real implementation, you'd use python-docx or similar
            return {
                'type': 'document',
                'format': file_path.suffix.lower()[1:]
            }
        except Exception as e:
            logger.debug(f"Failed to extract Office metadata: {e}")
        
        return {'type': 'document', 'format': file_path.suffix.lower()[1:]}
    
    @staticmethod
    def _get_gps_coords(gps_dict, coord_tag) -> Optional[str]:
        """Convert GPS coordinates to decimal format"""
        try:
            if coord_tag not in gps_dict:
                return None
            
            # Convert DMS to decimal
            dms = gps_dict[coord_tag]
            degrees = dms[0][0] / dms[0][1]
            minutes = dms[1][0] / dms[1][1]
            seconds = dms[2][0] / dms[2][1]
            
            decimal = degrees + minutes / 60 + seconds / 3600
            
            # Determine sign based on direction
            if coord_tag == piexif.GPSIFD.GPSLatitude:
                if piexif.GPSIFD.GPSLatitudeRef in gps_dict and gps_dict[piexif.GPSIFD.GPSLatitudeRef] == b'S':
                    decimal = -decimal
            elif coord_tag == piexif.GPSIFD.GPSLongitude:
                if piexif.GPSIFD.GPSLongitudeRef in gps_dict and gps_dict[piexif.GPSIFD.GPSLongitudeRef] == b'W':
                    decimal = -decimal
            
            return f"{decimal:.6f}"
        except Exception:
            return None

# ==========================================
# File Watcher for Real-time Organization
# ==========================================
class FileWatcher(FileSystemEventHandler):
    """Watch for file changes and organize them in real-time"""
    
    def __init__(self, organizer: 'SmartOrganizer'):
        self.organizer = organizer
        self.pending_files = set()
        self.timer = None
        self.delay = 5  # seconds to wait before organizing
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            self.pending_files.add(file_path)
            self._schedule_organization()
    
    def _schedule_organization(self):
        """Schedule organization after a delay to batch process files"""
        if self.timer:
            self.timer.cancel()
        
        self.timer = threading.Timer(self.delay, self._organize_pending_files)
        self.timer.start()
    
    def _organize_pending_files(self):
        """Organize all pending files"""
        if not self.pending_files:
            return
        
        logger.info(f"Organizing {len(self.pending_files)} new files...")
        
        files_info = []
        for file_path in list(self.pending_files):
            if file_path.exists():
                file_info = self.organizer._get_file_info(file_path)
                if file_info:
                    files_info.append(file_info)
            
            self.pending_files.discard(file_path)
        
        if files_info:
            self.organizer._organize_files(files_info)

# ==========================================
# Enhanced Smart Organizer Class
# ==========================================
class SmartOrganizer:
    """Enhanced file organizer with AI and metadata support"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config.load()
        self.target_dir = Path(self.config.TARGET_FOLDER)
        self.operations_log = []
        self.client = genai.Client(api_key=self.config.API_KEY)
        self.cache = ResponseCache(self.config.CACHE_DB)
        self.observer = None
        
        # Ensure target directory exists
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Start file watcher if enabled
        if self.config.ENABLE_WATCHER:
            self._start_watcher()
        
        # Schedule automatic organization if enabled
        if self.config.AUTO_ORGANIZE:
            self._schedule_auto_organize()
    
    def _start_watcher(self):
        """Start the file system watcher"""
        self.observer = Observer()
        event_handler = FileWatcher(self)
        self.observer.schedule(event_handler, str(self.target_dir), recursive=False)
        self.observer.start()
        logger.info("File watcher started")
    
    def _schedule_auto_organize(self):
        """Schedule automatic organization"""
        if self.config.ORGANIZE_INTERVAL == "hourly":
            schedule.every().hour.do(self.run)
        elif self.config.ORGANIZE_INTERVAL == "daily":
            schedule.every().day.at("02:00").do(self.run)
        elif self.config.ORGANIZE_INTERVAL == "weekly":
            schedule.every().week.do(self.run)
        
        logger.info(f"Auto-organization scheduled: {self.config.ORGANIZE_INTERVAL}")
        
        # Run scheduler in a separate thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def _get_file_info(self, file_path: Path) -> Optional[Dict]:
        """Get comprehensive information about a file"""
        if not file_path.is_file() or file_path.name.startswith('.'):
            return None
        
        try:
            stat = file_path.stat()
            file_info = {
                'name': file_path.name,
                'ext': file_path.suffix.lower(),
                'size_kb': round(stat.st_size / 1024, 2),
                'path': str(file_path),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
            }
            
            # Add file snippet for readable files
            if file_info['ext'] in self.config.READABLE_EXTS:
                file_info['snippet'] = self._get_file_snippet(file_path)
            
            # Add metadata for supported files
            if file_info['ext'] in self.config.METADATA_EXTS:
                file_info['metadata'] = MetadataExtractor.extract_metadata(file_path)
            
            return file_info
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return None
    
    def _get_file_snippet(self, file_path: Path) -> str:
        """Read a snippet from a file to help with categorization"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(1000).replace('\n', ' ')  # First 1000 characters
        except Exception:
            return ""
    
    def scan_directory(self) -> List[Dict]:
        """Scan the target directory and return file information"""
        files_info = []
        logger.info(f"Scanning folder: {self.target_dir}")
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = []
            
            for entry in self.target_dir.iterdir():
                futures.append(executor.submit(self._get_file_info, entry))
            
            for future in as_completed(futures):
                file_info = future.result()
                if file_info:
                    files_info.append(file_info)
        
        logger.info(f"Found {len(files_info)} files to organize")
        return files_info
    
    def ask_gemini(self, files_batch: List[Dict]) -> Dict[str, str]:
        """Send files to Gemini for categorization with caching"""
        # Create a hash for the batch request
        batch_content = json.dumps([f['name'] + f.get('snippet', '')[:200] for f in files_batch])
        request_hash = hashlib.md5(batch_content.encode()).hexdigest()
        
        # Check cache first
        cached_response = self.cache.get_ai_response(request_hash)
        if cached_response:
            logger.debug("Using cached AI response")
            return json.loads(cached_response)
        
        # Prepare files summary for the AI
        files_summary = []
        for f in files_batch:
            file_desc = f"File: {f['name']} | Size: {f['size_kb']}KB"
            
            # Add snippet if available
            if 'snippet' in f and f['snippet']:
                file_desc += f" | Context: {f['snippet'][:200]}"
            
            # Add metadata if available
            if 'metadata' in f and f['metadata']:
                metadata = f['metadata']
                if metadata.get('type'):
                    file_desc += f" | Type: {metadata['type']}"
                if metadata.get('camera'):
                    file_desc += f" | Camera: {metadata['camera']}"
                if metadata.get('date_taken'):
                    file_desc += f" | Date Taken: {metadata['date_taken']}"
                if metadata.get('location'):
                    file_desc += f" | Location: {metadata['location']}"
                if metadata.get('dimensions'):
                    file_desc += f" | Dimensions: {metadata['dimensions']}"
            
            files_summary.append(file_desc)
        
        prompt = f"""
        You are an intelligent file system administrator. 
        Categorize these files based on their Names, Extensions, Content, and Metadata.

        RULES:
        1. Create specific folders (e.g., 'Python Scripts', 'Financial Reports', 'Installers', 'Memes').
        2. Do NOT use generic names like 'Files' or 'Documents' unless necessary.
        3. For images, consider creating date-based folders (e.g., '2023-05-15 Trip to Paris').
        4. For documents with dates, create date-based folders (e.g., '2023-Q1 Reports').
        5. Return JSON ONLY: {{"filename.ext": "FolderName"}}
        
        FILES TO SORT:
        {json.dumps(files_summary)}
        """
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json'
                    )
                )
                
                result = json.loads(response.text)
                
                # Cache the response
                self.cache.cache_ai_response(request_hash, response.text)
                
                return result
            except Exception as e:
                wait_time = (attempt + 1) * 2
                logger.warning(f"API Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        logger.error("Failed to get AI response after retries.")
        return self._fallback_categorize(files_batch)
    
    def _fallback_categorize(self, files: List[Dict]) -> Dict[str, str]:
        """Categorize files using custom rules when AI fails"""
        logger.info("Using fallback categorization logic.")
        mapping = {}
        
        for f in files:
            folder = "Miscellaneous"
            
            # Check custom rules first
            for category, exts in self.config.CUSTOM_RULES.items():
                if f['ext'] in exts:
                    folder = category
                    break
            
            # Special handling for images with metadata
            if folder == "Images" and 'metadata' in f and f['metadata']:
                metadata = f['metadata']
                
                # Date-based folder for photos with date info
                if metadata.get('date_taken'):
                    try:
                        date_obj = datetime.strptime(metadata['date_taken'], "%Y:%m:%d %H:%M:%S")
                        folder = f"Photos/{date_obj.strftime('%Y-%m')}"
                    except:
                        pass
                
                # Location-based folder for photos with GPS info
                elif metadata.get('location'):
                    folder = f"Photos/Unknown Location"
            
            mapping[f['name']] = folder
        
        return mapping
    
    def _organize_files(self, files: List[Dict]):
        """Organize a list of files"""
        if not files:
            return
        
        # Show preview if enabled
        if self.config.SHOW_PREVIEW:
            if not self._show_preview(files):
                logger.info("Organization cancelled by user")
                return
        
        # Process files in batches
        for i in range(0, len(files), self.config.BATCH_SIZE):
            batch = files[i:i+self.config.BATCH_SIZE]
            logger.info(f"Processing batch {i//self.config.BATCH_SIZE + 1}...")
            
            # Check cache first
            categories = {}
            uncached_files = []
            
            for file_info in batch:
                cached_category = self.cache.get_file_category(file_info)
                if cached_category:
                    categories[file_info['name']] = cached_category
                else:
                    uncached_files.append(file_info)
            
            # Get categories for uncached files
            if uncached_files:
                new_categories = self.ask_gemini(uncached_files)
                categories.update(new_categories)
                
                # Cache the new categories
                for file_info in uncached_files:
                    if file_info['name'] in new_categories:
                        self.cache.cache_file_category(file_info, new_categories[file_info['name']])
            
            # Move files in parallel
            with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                futures = []
                
                for file_info in batch:
                    folder = categories.get(file_info['name'])
                    if folder:
                        futures.append(executor.submit(self._move_file, file_info, folder))
                
                # Wait for all moves to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error moving file: {e}")
        
        self.save_log()
        self._cleanup_empty_dirs()
        
        # Compress old files if enabled
        if self.config.COMPRESS_OLD_FILES:
            self._compress_old_files()
    
    def _show_preview(self, files: List[Dict]) -> bool:
        """Show a preview of the organization plan"""
        # This is a simplified version - in a real implementation, you'd show a GUI
        logger.info("=== ORGANIZATION PREVIEW ===")
        
        # Get categories for all files
        categories = {}
        for file_info in files:
            cached_category = self.cache.get_file_category(file_info)
            if cached_category:
                categories[file_info['name']] = cached_category
        
        uncached_files = [f for f in files if f['name'] not in categories]
        if uncached_files:
            new_categories = self.ask_gemini(uncached_files)
            categories.update(new_categories)
        
        # Group by category
        by_category = {}
        for file_name, category in categories.items():
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(file_name)
        
        # Display the plan
        for category, file_names in by_category.items():
            logger.info(f"  {category}/")
            for file_name in file_names[:5]:  # Show only first 5 files per category
                logger.info(f"    - {file_name}")
            if len(file_names) > 5:
                logger.info(f"    ... and {len(file_names) - 5} more files")
        
        # In a real implementation, you'd ask for confirmation here
        return True
    
    def _move_file(self, file_info: Dict, category: str):
        """Move a file to its category folder"""
        source_path = Path(file_info['path'])
        dest_folder = self.target_dir / category
        
        if not self.config.DRY_RUN:
            dest_folder.mkdir(parents=True, exist_ok=True)
        
        dest_path = dest_folder / file_info['name']
        
        # Handle name conflicts
        if dest_path.exists():
            # Check if it's a duplicate
            src_hash = self._get_file_hash(source_path)
            dest_hash = self._get_file_hash(dest_path)
            
            if src_hash == dest_hash:
                logger.warning(f"Duplicate found: {file_info['name']} (Skipping)")
                return
            
            # Rename if different content
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_folder / f"{stem}_{counter}{suffix}"
                counter += 1
        
        # Move the file
        if not self.config.DRY_RUN:
            try:
                shutil.move(str(source_path), str(dest_path))
                self.operations_log.append({
                    'filename': file_info['name'],
                    'old_path': str(source_path),
                    'new_path': str(dest_path),
                    'category': category,
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"[MOVED] {file_info['name']} -> {category}/")
            except Exception as e:
                logger.error(f"Failed to move {file_info['name']}: {e}")
        else:
            logger.info(f"[DRY RUN] Would move {file_info['name']} -> {category}/")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for duplicate detection"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except Exception:
            return "unknown"
    
    def _cleanup_empty_dirs(self):
        """Remove empty directories"""
        if self.config.DRY_RUN:
            return
        
        for entry in self.target_dir.iterdir():
            if entry.is_dir() and not any(entry.iterdir()):
                try:
                    entry.rmdir()
                    logger.info(f"Removed empty folder: {entry.name}")
                except:
                    pass
    
    def _compress_old_files(self):
        """Compress files older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=self.config.COMPRESS_DAYS)
        
        for entry in self.target_dir.rglob('*'):
            if entry.is_file():
                try:
                    file_time = datetime.fromtimestamp(entry.stat().st_mtime)
                    if file_time < cutoff_date:
                        # Create archive in the same directory
                        archive_name = entry.with_suffix(entry.suffix + '.zip')
                        if not archive_name.exists():
                            with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                zipf.write(entry, entry.name)
                            
                            if not self.config.DRY_RUN:
                                entry.unlink()
                                logger.info(f"Compressed old file: {entry.name}")
                            else:
                                logger.info(f"[DRY RUN] Would compress: {entry.name}")
                except Exception as e:
                    logger.error(f"Error compressing {entry.name}: {e}")
    
    def run(self):
        """Run the organization process"""
        files = self.scan_directory()
        if not files:
            logger.info("No files found to organize.")
            return
        
        logger.info(f"Found {len(files)} files. Starting AI organization...")
        self._organize_files(files)
    
    def save_log(self):
        """Save the operations log"""
        if not self.operations_log or self.config.DRY_RUN:
            return
        
        history = []
        if os.path.exists(self.config.LOG_FILE):
            try:
                with open(self.config.LOG_FILE, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                pass
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'operations': self.operations_log
        })
        
        with open(self.config.LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Operations log saved successfully.")
    
    def undo_last(self):
        """Undo the last organization operation"""
        if not os.path.exists(self.config.LOG_FILE):
            logger.warning("No log file found.")
            return
        
        try:
            with open(self.config.LOG_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            logger.error("Failed to read log file.")
            return
        
        if not history:
            logger.warning("History is empty.")
            return
        
        last_batch = history.pop()
        logger.info(f"Undoing operations from {last_batch['timestamp']}...")
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = []
            
            for op in last_batch['operations']:
                futures.append(executor.submit(self._restore_file, op))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error restoring file: {e}")
        
        with open(self.config.LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        self._cleanup_empty_dirs()
        logger.info("Undo completed.")
    
    def _restore_file(self, operation: Dict):
        """Restore a file to its original location"""
        try:
            if os.path.exists(operation['new_path']):
                os.makedirs(os.path.dirname(operation['old_path']), exist_ok=True)
                shutil.move(operation['new_path'], operation['old_path'])
                logger.info(f"Restored: {operation['filename']}")
        except Exception as e:
            logger.error(f"Failed to restore {operation['filename']}: {e}")
    
    def search_files(self, query: str, search_content: bool = False) -> List[Dict]:
        """Search for files by name or content"""
        results = []
        
        for entry in self.target_dir.rglob('*'):
            if entry.is_file():
                # Name search
                if query.lower() in entry.name.lower():
                    file_info = self._get_file_info(entry)
                    if file_info:
                        results.append(file_info)
                    continue
                
                # Content search
                if search_content and entry.suffix.lower() in self.config.READABLE_EXTS:
                    try:
                        with open(entry, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if query.lower() in content.lower():
                                file_info = self._get_file_info(entry)
                                if file_info:
                                    results.append(file_info)
                    except:
                        pass
        
        return results
    
    def stop(self):
        """Stop the organizer and clean up resources"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        # Clean up old cache entries
        self.cache.cleanup_old_entries()
        
        logger.info("Organizer stopped")

# ==========================================
# GUI for the Smart Organizer
# ==========================================
class OrganizerGUI:
    """GUI for the Smart Organizer"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart File Organizer")
        self.root.geometry("800x600")
        
        # Load or create config
        self.config = Config.load()
        
        # Create organizer
        self.organizer = SmartOrganizer(self.config)
        
        # Setup UI
        self._setup_ui()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Target folder selection
        folder_frame = ttk.Frame(main_frame)
        folder_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(folder_frame, text="Target Folder:").pack(side=tk.LEFT)
        
        self.folder_var = tk.StringVar(value=self.config.TARGET_FOLDER)
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=50)
        folder_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(folder_frame, text="Browse", command=self._browse_folder).pack(side=tk.LEFT)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=10)
        
        # Dry run checkbox
        self.dry_run_var = tk.BooleanVar(value=self.config.DRY_RUN)
        ttk.Checkbutton(options_frame, text="Dry Run (don't actually move files)", 
                       variable=self.dry_run_var).pack(anchor=tk.W)
        
        # Auto organize checkbox
        self.auto_organize_var = tk.BooleanVar(value=self.config.AUTO_ORGANIZE)
        ttk.Checkbutton(options_frame, text="Auto Organize", 
                       variable=self.auto_organize_var,
                       command=self._toggle_auto_organize).pack(anchor=tk.W)
        
        # Auto organize interval
        interval_frame = ttk.Frame(options_frame)
        interval_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(interval_frame, text="Auto Organize Interval:").pack(side=tk.LEFT)
        
        self.interval_var = tk.StringVar(value=self.config.ORGANIZE_INTERVAL)
        interval_combo = ttk.Combobox(interval_frame, textvariable=self.interval_var, 
                                     values=["hourly", "daily", "weekly"], state="readonly")
        interval_combo.pack(side=tk.LEFT, padx=5)
        
        # Enable watcher checkbox
        self.watcher_var = tk.BooleanVar(value=self.config.ENABLE_WATCHER)
        ttk.Checkbutton(options_frame, text="Enable Real-time File Watching", 
                       variable=self.watcher_var,
                       command=self._toggle_watcher).pack(anchor=tk.W)
        
        # Show preview checkbox
        self.preview_var = tk.BooleanVar(value=self.config.SHOW_PREVIEW)
        ttk.Checkbutton(options_frame, text="Show Preview Before Organizing", 
                       variable=self.preview_var).pack(anchor=tk.W)
        
        # Compress old files checkbox
        self.compress_var = tk.BooleanVar(value=self.config.COMPRESS_OLD_FILES)
        ttk.Checkbutton(options_frame, text="Compress Files Older Than 30 Days", 
                       variable=self.compress_var).pack(anchor=tk.W)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Organize Now", command=self._organize).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Undo Last", command=self._undo).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Search Files", command=self._search).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Save Config", command=self._save_config).pack(side=tk.LEFT, padx=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log text widget
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Redirect logging to the GUI
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging to display in the GUI"""
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)
        
        # Add GUI handler to logger
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(gui_handler)
    
    def _browse_folder(self):
        """Browse for target folder"""
        folder = filedialog.askdirectory(initialdir=self.folder_var.get())
        if folder:
            self.folder_var.set(folder)
    
    def _toggle_auto_organize(self):
        """Toggle auto organize option"""
        if self.auto_organize_var.get():
            self.organizer._schedule_auto_organize()
        else:
            schedule.clear()
    
    def _toggle_watcher(self):
        """Toggle file watcher option"""
        if self.watcher_var.get():
            self.organizer._start_watcher()
        else:
            if self.organizer.observer:
                self.organizer.observer.stop()
                self.organizer.observer.join()
                self.organizer.observer = None
    
    def _organize(self):
        """Run the organization process"""
        # Update config
        self._update_config()
        
        # Run in a separate thread to avoid freezing the GUI
        threading.Thread(target=self.organizer.run, daemon=True).start()
    
    def _undo(self):
        """Undo the last organization operation"""
        # Run in a separate thread to avoid freezing the GUI
        threading.Thread(target=self.organizer.undo_last, daemon=True).start()
    
    def _search(self):
        """Search for files"""
        # Create search dialog
        search_dialog = tk.Toplevel(self.root)
        search_dialog.title("Search Files")
        search_dialog.geometry("500x400")
        search_dialog.transient(self.root)
        search_dialog.grab_set()
        
        # Search frame
        search_frame = ttk.Frame(search_dialog, padding="10")
        search_frame.pack(fill=tk.X)
        
        ttk.Label(search_frame, text="Search Query:").pack(side=tk.LEFT)
        
        query_var = tk.StringVar()
        query_entry = ttk.Entry(search_frame, textvariable=query_var, width=30)
        query_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        search_content_var = tk.BooleanVar()
        ttk.Checkbutton(search_frame, text="Search in Content", 
                       variable=search_content_var).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.Frame(search_dialog)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Results treeview
        columns = ("Name", "Path", "Size", "Modified")
        results_tree = ttk.Treeview(results_frame, columns=columns, show="headings")
        
        for col in columns:
            results_tree.heading(col, text=col)
            results_tree.column(col, width=100)
        
        results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_tree.configure(yscrollcommand=scrollbar.set)
        
        # Search function
        def do_search():
            # Clear previous results
            for item in results_tree.get_children():
                results_tree.delete(item)
            
            # Perform search
            query = query_var.get()
            if not query:
                return
            
            search_content = search_content_var.get()
            results = self.organizer.search_files(query, search_content)
            
            # Add results to treeview
            for result in results:
                results_tree.insert("", tk.END, values=(
                    result['name'],
                    result['path'],
                    f"{result['size_kb']} KB",
                    result['modified']
                ))
        
        # Search button
        ttk.Button(search_dialog, text="Search", command=do_search).pack(pady=10)
    
    def _save_config(self):
        """Save the current configuration"""
        self._update_config()
        self.config.save()
        messagebox.showinfo("Config Saved", "Configuration saved successfully.")
    
    def _update_config(self):
        """Update the config with current GUI values"""
        self.config.TARGET_FOLDER = self.folder_var.get()
        self.config.DRY_RUN = self.dry_run_var.get()
        self.config.AUTO_ORGANIZE = self.auto_organize_var.get()
        self.config.ORGANIZE_INTERVAL = self.interval_var.get()
        self.config.ENABLE_WATCHER = self.watcher_var.get()
        self.config.SHOW_PREVIEW = self.preview_var.get()
        self.config.COMPRESS_OLD_FILES = self.compress_var.get()
        
        # Update organizer config
        self.organizer.config = self.config
    
    def _on_closing(self):
        """Handle window closing event"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.organizer.stop()
            self.root.destroy()

# ==========================================
# Main Entry Point
# ==========================================
def main():
    """Main entry point"""
    import sys
    
    # Check if GUI mode
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        root = tk.Tk()
        app = OrganizerGUI(root)
        root.mainloop()
        return
    
    # CLI mode
    config = Config.load()
    organizer = SmartOrganizer(config)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'undo':
            organizer.undo_last()
        elif sys.argv[1] == 'search':
            if len(sys.argv) > 2:
                query = ' '.join(sys.argv[2:])
                results = organizer.search_files(query)
                for result in results:
                    print(f"{result['path']} ({result['size_kb']} KB)")
            else:
                print("Please provide a search query")
        else:
            print(f"Unknown command: {sys.argv[1]}")
    else:
        organizer.run()

if __name__ == "__main__":
    main()
