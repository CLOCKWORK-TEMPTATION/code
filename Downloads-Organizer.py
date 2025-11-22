"""
منظم التنزيلات الذكي - الإصدار المحسن

يوفر هذا البرنامج نظامًا متقدمًا لتنظيم الملفات يستخدم الذكاء الاصطناعي (Google Gemini)،
والتعلم الآلي (التجميع)، وتقنيات أخرى متنوعة لفرز وإعادة تسمية وإدارة الملفات تلقائيًا
في دليل مستهدف (الافتراضي هو التنزيلات).

الميزات:
    - تصنيف مدعوم بالذكاء الاصطناعي باستخدام Google Gemini.
    - تحليل المحتوى المحلي واستخراج النص.
    - تجميع التعلم الآلي لتجميع الملفات المتشابهة.
    - اكتشاف التكرارات (المطابقة التامة وشبه المطابقة للصور).
    - دعم النسخ الاحتياطي السحابي (AWS S3).
    - واجهة برمجة تطبيقات REST للإدارة عن بعد.
    - واجهات المستخدم الرسومية (GUI) وسطر الأوامر (CLI).
    - إصدار الملفات وإعادة التسمية الذكية.
"""

import os
import sys
import shutil
import json
import time
import hashlib
import logging
import asyncio
import sqlite3
import schedule
import platform
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas
import threading
import webbrowser
from collections import defaultdict

# Optional imports
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

try:
    from PIL import Image, ImageTk, ImageDraw
    import piexif
    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False

try:
    from aiohttp import web
    import aiohttp_cors
    HAS_WEB = True
except ImportError:
    HAS_WEB = False

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_ML = True
except ImportError:
    HAS_ML = False

try:
    import boto3
    HAS_S3 = True
except ImportError:
    HAS_S3 = False

try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ==========================================
# CONFIGURATION
# ==========================================

class DuplicateStrategy(Enum):
    """
    تعداد لاستراتيجيات التعامل مع الملفات المكررة.

    السمات:
        SKIP: تخطي الملف المكرر.
        RENAME: إعادة تسمية الملف الجديد بلاحقة عداد.
        MOVE: نقل الملف المكرر إلى موقع نسخ احتياطي.
        DELETE: حذف الملف المكرر.
        SMART_MERGE: الاحتفاظ بأحدث ملف ونقل الآخرين إلى النسخ الاحتياطي.
        VERSION: الاحتفاظ بكلا الملفين ولكن تعليم أحدهما كنسخة من الآخر.
    """
    SKIP = "skip"
    RENAME = "rename"
    MOVE = "move"
    DELETE = "delete"
    SMART_MERGE = "smart_merge"
    VERSION = "version"  # New strategy: keep both versions

class CloudProvider(Enum):
    """
    تعداد لمزودي خدمات التخزين السحابي المدعومين.

    السمات:
        NONE: لا يوجد نسخ احتياطي سحابي.
        AWS_S3: خدمات أمازون ويب S3.
        GOOGLE_DRIVE: جوجل درايف (محجوز).
        ONEDRIVE: مايكروسوفت ون درايف (محجوز).
    """
    NONE = "none"
    AWS_S3 = "aws_s3"
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"

@dataclass
class Config:
    """
    فئة التكوين للتطبيق.

    تخزن جميع الإعدادات المتعلقة بالمسارات، وتكوين الذكاء الاصطناعي، وخيارات المعالجة،
    وتبديل الميزات.
    """
    # Paths
    target_folder: str = str(Path.home() / "Downloads")
    organized_root: str = ""
    cache_db: str = ""
    backup_folder: str = ""
    
    # AI
    api_key: str = "YOUR_GEMINI_API_KEY"
    model: str = "gemini-2.0-flash-exp"
    enable_content_analysis: bool = True
    
    # Processing
    batch_size: int = 50
    max_workers: int = 8
    dry_run: bool = False
    
    # Features
    smart_rename: bool = True
    enable_watcher: bool = False
    cleanup_junk: bool = True
    duplicate_strategy: str = "smart_merge"
    enable_gui: bool = True
    enable_rest_api: bool = False
    rest_api_port: int = 8080
    enable_ml_clustering: bool = False
    enable_versioning: bool = True
    enable_cloud_backup: bool = False
    cloud_provider: str = "none"
    cloud_settings: Dict = field(default_factory=dict)
    enable_scheduling: bool = False
    schedule_interval: str = "daily"  # hourly, daily, weekly
    schedule_time: str = "02:00"  # HH:MM format
    enable_content_search: bool = True
    enable_file_preview: bool = True
    
    # Security
    allowed_extensions: Set[str] = field(default_factory=lambda: {
        '.txt', '.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png', '.gif',
        '.mp4', '.mp3', '.zip', '.rar', '.py', '.js', '.html', '.css', 
        '.xlsx', '.pptx', '.svg', '.webp', '.mkv', '.mov', '.avi', '.wav',
        '.flac', '.aac', '.7z', '.tar', '.gz', '.java', '.cpp', '.c', '.h'
    })
    
    custom_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp', '.tiff'],
        'Videos': ['.mp4', '.mkv', '.mov', '.avi', '.wmv', '.flv', '.webm'],
        'Documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
        'Spreadsheets': ['.xlsx', '.xls', '.csv', '.ods'],
        'Presentations': ['.pptx', '.ppt', '.odp'],
        'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.php', '.rb'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'],
        'Ebooks': ['.epub', '.mobi', '.azw', '.azw3'],
        'Fonts': ['.ttf', '.otf', '.woff', '.woff2']
    })
    
    def __post_init__(self):
        """
        ما بعد التهيئة لتعيين المسارات المشتقة الافتراضية إذا لم يتم توفيرها.
        يضمن وجود الدلائل المستهدفة.
        """
        if not self.organized_root:
            self.organized_root = str(Path(self.target_folder) / "Organized")
        if not self.cache_db:
            self.cache_db = str(Path(self.target_folder) / ".organizer_cache.db")
        if not self.backup_folder:
            self.backup_folder = str(Path(self.target_folder) / "Backup")
            
        # Create necessary directories
        Path(self.organized_root).mkdir(parents=True, exist_ok=True)
        Path(self.backup_folder).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load(cls, path: str = None) -> 'Config':
        """
        تحميل التكوين من ملف JSON.

        المعاملات:
            path (str, optional): مسار ملف التكوين. الافتراضي هو ~/Downloads/.organizer_config.json.

        القيمة المرجعة:
            Config: كائن Config معبأ بالبيانات.
        """
        if path is None:
            path = Path.home() / "Downloads" / ".organizer_config.json"
        
        if Path(path).exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
            except Exception:
                pass
        return cls()
    
    def save(self, path: str = None):
        """
        حفظ التكوين الحالي في ملف JSON.

        المعاملات:
            path (str, optional): مسار حفظ ملف التكوين. الافتراضي هو .organizer_config.json في المجلد المستهدف.
        """
        if path is None:
            path = Path(self.target_folder) / ".organizer_config.json"
        
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)

# ==========================================
# LOGGING
# ==========================================

def setup_logging(config: Config) -> logging.Logger:
    """
    يقوم بإعداد تكوين السجلات (Logging) للتطبيق.

    يتم إخراج السجلات إلى كل من وحدة التحكم (Console) وملف باسم 'organizer.log' في المجلد المستهدف.

    المعاملات:
        config (Config): كائن التكوين الذي يحتوي على مسار المجلد المستهدف.

    القيمة المرجعة:
        logging.Logger: نسخة مهيأة من المسجل.
    """
    logger = logging.getLogger("OrganizerUltimate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(console)
    
    # File handler
    log_file = Path(config.target_folder) / "organizer.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

logger = None

# ==========================================
# DATABASE
# ==========================================

class Database:
    """
    يدير تفاعلات قاعدة بيانات SQLite لتخزين البيانات الوصفية للملفات، وتاريخ العمليات،
    وفهرسة البحث في النص الكامل.
    """
    def __init__(self, db_path: str):
        """
        تهيئة معالج اتصال قاعدة البيانات.

        المعاملات:
            db_path (str): مسار الملف لقاعدة بيانات SQLite.
        """
        self.db_path = db_path
        self.conn = None
    
    def init(self):
        """
        تهيئة مخطط قاعدة البيانات، وإنشاء الجداول والفهارس إذا لم تكن موجودة.
        """
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Files table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                name TEXT,
                hash TEXT,
                category TEXT,
                new_name TEXT,
                organized_at DATETIME,
                metadata TEXT,
                content_text TEXT,
                version INTEGER DEFAULT 1,
                cloud_backup BOOLEAN DEFAULT 0
            )
        """)
        
        # Operations table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY,
                operation_type TEXT,
                file_path TEXT,
                dest_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                undone BOOLEAN DEFAULT 0
            )
        """)
        
        # File versions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_versions (
                id INTEGER PRIMARY KEY,
                file_hash TEXT,
                version_path TEXT,
                version_number INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_hash) REFERENCES files (hash)
            )
        """)
        
        # Content search index
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS content_index USING fts5(
                file_id,
                content_text,
                file_name
            )
        """)
        
        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON files(hash)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON files(category)")
        self.conn.commit()
    
    def get_category(self, file_hash: str, file_name: str) -> Optional[Dict]:
        """
        استرجاع معلومات الفئة المخزنة مؤقتًا لملف بناءً على التجزئة (Hash) والاسم.

        المعاملات:
            file_hash (str): تجزئة SHA-256 للملف.
            file_name (str): اسم الملف.

        القيمة المرجعة:
            Optional[Dict]: قاموس يحتوي على 'folder' و 'new_name' إذا وجد، وإلا None.
        """
        cursor = self.conn.execute(
            "SELECT category, new_name FROM files WHERE hash = ? AND name = ?",
            (file_hash, file_name)
        )
        result = cursor.fetchone()
        if result:
            return {'folder': result['category'], 'new_name': result['new_name']}
        return None
    
    def cache_category(self, file_info: Dict, category_data: Dict):
        """
        تخزين نتيجة التصنيف والبيانات الوصفية للملف مؤقتًا.

        المعاملات:
            file_info (Dict): معلومات حول الملف (المسار، الاسم، التجزئة، إلخ).
            category_data (Dict): الفئة المحددة والاسم الجديد المحتمل.
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO files (path, name, hash, category, new_name, organized_at, metadata, content_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_info['path'],
            file_info['name'],
            file_info['hash'],
            category_data.get('folder'),
            category_data.get('new_name'),
            datetime.now().isoformat(),
            json.dumps(file_info.get('metadata', {})),
            file_info.get('content_text', '')
        ))
        
        # Update content search index
        file_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute("""
            INSERT OR REPLACE INTO content_index (file_id, content_text, file_name)
            VALUES (?, ?, ?)
        """, (file_id, file_info.get('content_text', ''), file_info['name']))
        
        self.conn.commit()
    
    def log_operation(self, op_type: str, file_path: str, dest_path: str):
        """
        تسجيل عملية ملف (نقل، إعادة تسمية، إلخ) لمسارات التدقيق ووظيفة التراجع.

        المعاملات:
            op_type (str): نوع العملية (مثل 'move').
            file_path (str): المسار الأصلي للملف.
            dest_path (str): المسار الوجهة للملف.
        """
        self.conn.execute("""
            INSERT INTO operations (operation_type, file_path, dest_path)
            VALUES (?, ?, ?)
        """, (op_type, file_path, dest_path))
        self.conn.commit()
    
    def get_recent_operations(self, limit: int = 10) -> List[Dict]:
        """
        استرجاع أحدث العمليات التي لم يتم التراجع عنها.

        المعاملات:
            limit (int, optional): الحد الأقصى لعدد العمليات المراد إرجاعها. الافتراضي هو 10.

        القيمة المرجعة:
            List[Dict]: قائمة بسجلات العمليات.
        """
        cursor = self.conn.execute("""
            SELECT * FROM operations WHERE undone = 0 
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def search_content(self, query: str) -> List[Dict]:
        """
        إجراء بحث في النص الكامل على محتوى الملف وأسمائه.

        المعاملات:
            query (str): نص البحث.

        القيمة المرجعة:
            List[Dict]: قائمة بسجلات الملفات المطابقة.
        """
        cursor = self.conn.execute("""
            SELECT f.* FROM files f
            JOIN content_index ci ON f.id = ci.file_id
            WHERE content_index MATCH ?
            ORDER BY rank
        """, (query,))
        return [dict(row) for row in cursor.fetchall()]
    
    def add_file_version(self, file_hash: str, version_path: str, version_number: int):
        """
        تسجيل نسخة جديدة من الملف.

        المعاملات:
            file_hash (str): تجزئة محتوى الملف الأصلي.
            version_path (str): المسار الذي تم تخزين النسخة فيه.
            version_number (int): رقم تسلسل النسخة.
        """
        self.conn.execute("""
            INSERT INTO file_versions (file_hash, version_path, version_number)
            VALUES (?, ?, ?)
        """, (file_hash, version_path, version_number))
        self.conn.commit()
    
    def get_file_versions(self, file_hash: str) -> List[Dict]:
        """
        استرجاع كل تاريخ الإصدارات لتجزئة محتوى ملف محدد.

        المعاملات:
            file_hash (str): تجزئة الملف.

        القيمة المرجعة:
            List[Dict]: قائمة بسجلات الإصدارات مرتبة حسب رقم الإصدار تنازليًا.
        """
        cursor = self.conn.execute("""
            SELECT * FROM file_versions WHERE file_hash = ? ORDER BY version_number DESC
        """, (file_hash,))
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """إغلاق اتصال قاعدة البيانات."""
        if self.conn:
            self.conn.close()

# ==========================================
# FILE ANALYZER
# ==========================================

class FileAnalyzer:
    """
    يقوم بتحليل الملفات لاستخراج البيانات الوصفية، وحساب التجزئة (Hashes)، وإنشاء معاينات.
    """
    def __init__(self, config: Config):
        """
        تهيئة المحلل.

        المعاملات:
            config (Config): كائن التكوين.
        """
        self.config = config
    
    async def analyze(self, file_path: Path) -> Dict:
        """
        تحليل ملف واحد لجمع بيانات وصفية شاملة.

        المعاملات:
            file_path (Path): مسار الملف.

        القيمة المرجعة:
            Dict: قاموس يحتوي على معلومات الملف (التجزئة، الحجم، البيانات الوصفية، نص المحتوى، إلخ).
        """
        info = {
            'name': file_path.name,
            'path': str(file_path),
            'ext': file_path.suffix.lower(),
            'size_kb': 0,
            'hash': '',
            'modified': '',
            'metadata': {},
            'content_text': '',
            'preview_path': ''
        }
        
        try:
            stat = file_path.stat()
            info['size_kb'] = round(stat.st_size / 1024, 2)
            info['modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            info['hash'] = await self._calc_hash(file_path)
            
            # Extract metadata based on file type
            if HAS_IMAGE and info['ext'] in {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}:
                info['metadata'] = await self._extract_image_meta(file_path)
                # Generate thumbnail
                if self.config.enable_file_preview:
                    info['preview_path'] = await self._generate_thumbnail(file_path)
            
            # Extract text content for search
            if self.config.enable_content_search:
                info['content_text'] = await self._extract_text_content(file_path)
                
        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
        
        return info
    
    async def _calc_hash(self, file_path: Path) -> str:
        """
        حساب تجزئة SHA-256 للملف.

        المعاملات:
            file_path (Path): مسار الملف.

        القيمة المرجعة:
            str: النص الست عشرية للتجزئة.
        """
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except:
            return "unknown"
    
    async def _extract_image_meta(self, file_path: Path) -> Dict:
        """
        استخراج البيانات الوصفية للصورة (الأبعاد، بيانات EXIF).

        المعاملات:
            file_path (Path): مسار ملف الصورة.

        القيمة المرجعة:
            Dict: قاموس البيانات الوصفية.
        """
        meta = {}
        try:
            with Image.open(file_path) as img:
                meta['dimensions'] = f"{img.width}x{img.height}"
                meta['format'] = img.format
            
            if file_path.suffix.lower() in {'.jpg', '.jpeg'}:
                exif_dict = piexif.load(str(file_path))
                if '0th' in exif_dict and piexif.ImageIFD.Make in exif_dict['0th']:
                    make = exif_dict['0th'][piexif.ImageIFD.Make].decode('utf-8', errors='ignore')
                    meta['camera'] = make
                
                if 'Exif' in exif_dict and piexif.ExifIFD.DateTimeOriginal in exif_dict['Exif']:
                    date_taken = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8', errors='ignore')
                    meta['date_taken'] = date_taken
        except:
            pass
        
        return meta
    
    async def _generate_thumbnail(self, file_path: Path) -> str:
        """
        إنشاء صورة مصغرة لملف صورة.

        المعاملات:
            file_path (Path): مسار الصورة.

        القيمة المرجعة:
            str: مسار ملف الصورة المصغرة الذي تم إنشاؤه، أو سلسلة فارغة عند الفشل.
        """
        if not HAS_IMAGE:
            return ""
            
        try:
            thumb_dir = Path(self.config.target_folder) / ".thumbnails"
            thumb_dir.mkdir(exist_ok=True)
            
            thumb_path = thumb_dir / f"{file_path.stem}_thumb.jpg"
            
            # Skip if thumbnail already exists and is newer than the original
            if thumb_path.exists() and thumb_path.stat().st_mtime > file_path.stat().st_mtime:
                return str(thumb_path)
            
            with Image.open(file_path) as img:
                img.thumbnail((200, 200))
                img.convert('RGB').save(thumb_path, 'JPEG')
            
            return str(thumb_path)
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for {file_path}: {e}")
            return ""
    
    async def _extract_text_content(self, file_path: Path) -> str:
        """
        استخراج محتوى النص من أنواع ملفات مختلفة (txt, pdf, docx, image OCR).

        المعاملات:
            file_path (Path): مسار الملف.

        القيمة المرجعة:
            str: محتوى النص المستخرج.
        """
        content = ""
        
        try:
            # Text files
            if file_path.suffix.lower() in {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv'}:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:5000]  # Limit to first 5000 chars
            
            # PDF files
            elif file_path.suffix.lower() == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages[:3]:  # First 3 pages
                            content += page.extract_text()
                except ImportError:
                    logger.warning("PyPDF2 not installed, cannot extract text from PDFs")
                except Exception as e:
                    logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            
            # OCR for images
            elif HAS_OCR and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}:
                try:
                    with Image.open(file_path) as img:
                        content = pytesseract.image_to_string(img)
                except Exception as e:
                    logger.error(f"OCR failed for {file_path}: {e}")
            
            # Word documents
            elif file_path.suffix.lower() in {'.docx', '.doc'}:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    content = "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    logger.warning("python-docx not installed, cannot extract text from Word documents")
                except Exception as e:
                    logger.error(f"Failed to extract text from Word document {file_path}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
        
        return content

# ==========================================
# ADVANCED DUPLICATE DETECTOR
# ==========================================

class AdvancedDuplicateDetector:
    """
    اكتشاف الملفات المكررة باستخدام التجزئة الصارمة والمطابقة التقريبية/الاستدلالية للوسائط.
    """
    def __init__(self, config: Config, db: Database):
        """
        تهيئة الكاشف.

        المعاملات:
            config (Config): كائن التكوين.
            db (Database): نسخة قاعدة البيانات.
        """
        self.config = config
        self.db = db
        self.similarity_threshold = 0.85
    
    async def find_duplicates(self, files: List[Dict]) -> List[List[Dict]]:
        """
        تحديد مجموعات الملفات المكررة.

        المعاملات:
            files (List[Dict]): قائمة بقواميس معلومات الملفات.

        القيمة المرجعة:
            List[List[Dict]]: قائمة بقوائم، حيث تحتوي كل قائمة داخلية على إدخالات ملفات مكررة.
        """
        # Group by exact hash first
        hash_groups = defaultdict(list)
        for file_info in files:
            hash_groups[file_info['hash']].append(file_info)
        
        duplicates = []
        
        # Exact duplicates
        for file_hash, group in hash_groups.items():
            if len(group) > 1:
                duplicates.append(group)
        
        # For images, also check for near-duplicates using perceptual hashing
        if HAS_IMAGE:
            image_files = [f for f in files if f['ext'] in {'.jpg', '.jpeg', '.png'}]
            near_duplicates = await self._find_near_duplicate_images(image_files)
            duplicates.extend(near_duplicates)
        
        return duplicates
    
    async def _find_near_duplicate_images(self, image_files: List[Dict]) -> List[List[Dict]]:
        """
        العثور على الصور التي من المحتمل أن تكون مكررة بناءً على الاستدلال (الأبعاد، الحجم).

        المعاملات:
            image_files (List[Dict]): قائمة بمعلومات ملفات الصور.

        القيمة المرجعة:
            List[List[Dict]]: مجموعات من الصور المحتمل تكرارها.
        """
        # This is a simplified implementation
        # In a real scenario, you'd use more sophisticated image similarity algorithms
        # like perceptual hashing or feature matching
        
        # Group by similar dimensions and file size as a heuristic
        groups = defaultdict(list)
        for file_info in image_files:
            dimensions = file_info.get('metadata', {}).get('dimensions', '0x0')
            size = int(file_info['size_kb'])
            # Round size to nearest 10KB for grouping
            size_key = (dimensions, (size // 10) * 10)
            groups[size_key].append(file_info)
        
        duplicates = []
        for key, group in groups.items():
            if len(group) > 1:
                # For a more accurate approach, you'd calculate image similarity here
                duplicates.append(group)
        
        return duplicates
    
    async def handle_duplicates(self, duplicate_groups: List[List[Dict]]) -> List[Dict]:
        """
        تحديد الإجراءات التي يجب اتخاذها لكل مجموعة مكررة بناءً على الاستراتيجية المكونة.

        المعاملات:
            duplicate_groups (List[List[Dict]]): مجموعات التكرارات.

        القيمة المرجعة:
            List[Dict]: قائمة بكائنات الإجراء التي تحدد ما يجب فعله بملفات معينة.
        """
        actions = []
        
        for group in duplicate_groups:
            if self.config.duplicate_strategy == DuplicateStrategy.SKIP.value:
                # Keep the first one, mark others to skip
                for i, file_info in enumerate(group):
                    if i > 0:
                        actions.append({
                            'file': file_info,
                            'action': 'skip',
                            'reason': 'duplicate'
                        })
            
            elif self.config.duplicate_strategy == DuplicateStrategy.VERSION.value:
                # Keep all but version them
                for i, file_info in enumerate(group):
                    if i > 0:
                        actions.append({
                            'file': file_info,
                            'action': 'version',
                            'reason': 'duplicate',
                            'reference': group[0]
                        })
            
            elif self.config.duplicate_strategy == DuplicateStrategy.SMART_MERGE.value:
                # Keep the newest file based on modification date
                newest = max(group, key=lambda f: f['modified'])
                for file_info in group:
                    if file_info != newest:
                        actions.append({
                            'file': file_info,
                            'action': 'move_to_backup',
                            'reason': 'duplicate',
                            'reference': newest
                        })
        
        return actions

# ==========================================
# CLOUD BACKUP MANAGER
# ==========================================

class CloudBackupManager:
    """
    يدير تحميل الملفات إلى مزودي التخزين السحابي (حاليًا AWS S3).
    """
    def __init__(self, config: Config):
        """
        تهيئة مدير النسخ الاحتياطي السحابي.

        المعاملات:
            config (Config): كائن التكوين مع إعدادات السحابة.
        """
        self.config = config
        self.enabled = config.enable_cloud_backup and config.cloud_provider != CloudProvider.NONE.value
        self.client = None
        
        if self.enabled:
            self._initialize_client()
    
    def _initialize_client(self):
        """تهيئة عميل مزود السحابة (مثل boto3 لـ S3)."""
        if self.config.cloud_provider == CloudProvider.AWS_S3.value and HAS_S3:
            try:
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=self.config.cloud_settings.get('aws_access_key'),
                    aws_secret_access_key=self.config.cloud_settings.get('aws_secret_key'),
                    region_name=self.config.cloud_settings.get('aws_region', 'us-east-1')
                )
                
                # Create bucket if it doesn't exist
                bucket_name = self.config.cloud_settings.get('s3_bucket', 'file-organizer-backup')
                try:
                    self.client.head_bucket(Bucket=bucket_name)
                except:
                    self.client.create_bucket(Bucket=bucket_name)
                
                logger.info(f"Connected to AWS S3 bucket: {bucket_name}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                self.enabled = False
    
    async def backup_file(self, file_path: Path, category: str) -> Optional[str]:
        """
        نسخ احتياطي لملف واحد إلى السحابة.

        المعاملات:
            file_path (Path): مسار الملف.
            category (str): مجلد الفئة (يستخدم كبادئة/مجلد في السحابة).

        القيمة المرجعة:
            Optional[str]: معرف الموارد الموحد (URI) لـ S3 أو موقع السحابة إذا نجح، وإلا None.
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            # Create a unique key for the file
            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
            key = f"{category}/{file_path.name}_{file_hash}"
            
            if self.config.cloud_provider == CloudProvider.AWS_S3.value:
                bucket_name = self.config.cloud_settings.get('s3_bucket', 'file-organizer-backup')
                self.client.upload_file(str(file_path), bucket_name, key)
                return f"s3://{bucket_name}/{key}"
            
        except Exception as e:
            logger.error(f"Failed to backup {file_path} to cloud: {e}")
        
        return None

# ==========================================
# AI CATEGORIZER
# ==========================================

class AICategorizer:
    """
    يستخدم الذكاء الاصطناعي (Google Gemini) أو المنطق الاحتياطي لتصنيف الملفات.
    """
    def __init__(self, config: Config, db: Database):
        """
        تهيئة المصنف.

        المعاملات:
            config (Config): كائن التكوين (يحتاج إلى api_key).
            db (Database): نسخة قاعدة البيانات للتخزين المؤقت.
        """
        self.config = config
        self.db = db
        self.client = None
        
        if HAS_GEMINI and config.api_key != "YOUR_GEMINI_API_KEY":
            self.client = genai.Client(api_key=config.api_key)
    
    async def categorize_batch(self, files: List[Dict]) -> Dict[str, Dict]:
        """
        تصنيف دفعة من الملفات. يتحقق من ذاكرة التخزين المؤقت أولاً، ثم يستدعي الذكاء الاصطناعي إذا لزم الأمر.

        المعاملات:
            files (List[Dict]): قائمة بقواميس معلومات الملفات.

        القيمة المرجعة:
            Dict[str, Dict]: تعيين مسار الملف إلى بيانات الفئة ('folder', 'new_name').
        """
        results = {}
        uncached = []
        
        for f in files:
            cached = self.db.get_category(f['hash'], f['name'])
            if cached:
                results[f['path']] = cached
            else:
                uncached.append(f)
        
        if uncached:
            if self.client:
                ai_results = await self._ai_categorize(uncached)
            else:
                ai_results = self._fallback_categorize(uncached)
            
            results.update(ai_results)
            
            for f in uncached:
                if f['path'] in ai_results:
                    self.db.cache_category(f, ai_results[f['path']])
        
        return results
    
    async def _ai_categorize(self, files: List[Dict]) -> Dict[str, Dict]:
        """
        استدعاء واجهة برمجة تطبيقات Google Gemini لتصنيف الملفات.

        المعاملات:
            files (List[Dict]): قائمة بمعلومات الملفات.

        القيمة المرجعة:
            Dict[str, Dict]: نتائج التصنيف.
        """
        # Start with fallback results to ensure full coverage and handle failures gracefully
        results = self._fallback_categorize(files)
        prompt = self._build_prompt(files)
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    temperature=0.1
                )
            )
            response_data = json.loads(response.text)

            # Map AI response back to file paths using the generated IDs
            for i, f in enumerate(files):
                fid = str(i)
                if fid in response_data:
                    results[f['path']] = response_data[fid]

            return results
        except Exception as e:
            logger.error(f"AI failed: {e}")
            return results
    
    def _build_prompt(self, files: List[Dict]) -> str:
        """
        بناء الموجه (Prompt) لنموذج الذكاء الاصطناعي.

        المعاملات:
            files (List[Dict]): قائمة بمعلومات الملفات.

        القيمة المرجعة:
            str: سلسلة الموجه التي تم بناؤها.
        """
        files_json = []
        
        for i, f in enumerate(files):
            file_data = {
                'id': str(i),
                'name': f['name'],
                'ext': f['ext'],
                'size_kb': f['size_kb'],
                'metadata': f.get('metadata', {})
            }
            
            # Include content text for better categorization if enabled
            if self.config.enable_content_analysis and f.get('content_text'):
                file_data['content_preview'] = f['content_text'][:500]  # First 500 chars
            
            files_json.append(file_data)
        
        return f"""Categorize these files intelligently based on their names, extensions, metadata, and content.
Return JSON: {{"id": {{"folder": "FolderName", "new_name": "OptionalNewName.ext"}}}}

FILES:
{json.dumps(files_json, indent=2)}

Respond with ONLY the JSON."""
    
    def _fallback_categorize(self, files: List[Dict]) -> Dict[str, Dict]:
        """
        منطق التصنيف الاحتياطي بناءً على الامتدادات والكلمات الرئيسية إذا لم يكن الذكاء الاصطناعي متاحًا.

        المعاملات:
            files (List[Dict]): قائمة بمعلومات الملفات.

        القيمة المرجعة:
            Dict[str, Dict]: نتائج التصنيف.
        """
        results = {}
        for f in files:
            category = "Others"
            for cat, exts in self.config.custom_categories.items():
                if f['ext'] in exts:
                    category = cat
                    break
            
            # Try to infer category from filename patterns
            if category == "Others":
                name_lower = f['name'].lower()
                if any(term in name_lower for term in ['invoice', 'receipt', 'bill']):
                    category = "Finance"
                elif any(term in name_lower for term in ['contract', 'agreement', 'legal']):
                    category = "Legal"
                elif any(term in name_lower for term in ['resume', 'cv', 'application']):
                    category = "Career"
                elif any(term in name_lower for term in ['manual', 'guide', 'tutorial']):
                    category = "Documentation"
            
            results[f['path']] = {'folder': category, 'new_name': None}
        return results

# ==========================================
# ML CLUSTERING
# ==========================================

class MLClustering:
    """
    يستخدم التعلم الآلي (scikit-learn) لتجميع الملفات بناءً على ميزات النص.
    """
    def __init__(self, config: Config):
        """
        تهيئة تجميع التعلم الآلي.

        المعاملات:
            config (Config): كائن التكوين.
        """
        self.config = config
        self.enabled = HAS_ML and config.enable_ml_clustering
    
    def cluster_files(self, files: List[Dict], n_clusters: int = 5) -> Dict[int, List[Dict]]:
        """
        تجميع الملفات في مجموعات باستخدام TF-IDF و DBSCAN.

        المعاملات:
            files (List[Dict]): قائمة بمعلومات الملفات.
            n_clusters (int, optional): العدد المستهدف للمجموعات (لا يستخدم بصرامة بواسطة DBSCAN).

        القيمة المرجعة:
            Dict[int, List[Dict]]: قاموس يعين معرف المجموعة إلى قائمة الملفات.
        """
        if not self.enabled or not files:
            return {0: files}
        
        try:
            # Extract features from filenames and content
            features_text = []
            for f in files:
                text = f['name']
                if f.get('content_text'):
                    text += " " + f['content_text'][:500]  # Add content preview
                features_text.append(text)
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            features = vectorizer.fit_transform(features_text)
            
            # Use DBSCAN for better clustering of varying density
            clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
            labels = clustering.fit_predict(features.toarray())
            
            # Group by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[label].append(files[i])
            
            # If all points are noise (-1), put them in cluster 0
            if -1 in clusters and len(clusters) == 1:
                clusters = {0: files}
            
            logger.info(f"📊 ML Clustering: {len(files)} files → {len(clusters)} clusters")
            return dict(clusters)
        
        except Exception as e:
            logger.error(f"ML clustering failed: {e}")
            return {0: files}

# ==========================================
# REST API
# ==========================================

class RestAPI:
    """
    يوفر واجهة برمجة تطبيقات RESTful للتفاعل مع المنظم.
    """
    def __init__(self, organizer: 'SmartOrganizer', port: int = 8080):
        """
        تهيئة واجهة برمجة تطبيقات REST.

        المعاملات:
            organizer (SmartOrganizer): نسخة المنظم الرئيسية.
            port (int, optional): المنفذ للاستماع عليه. الافتراضي هو 8080.
        """
        self.organizer = organizer
        self.port = port
        self.app = None
        self.runner = None
    
    async def start(self):
        """بدء تشغيل خادم الويب aiohttp."""
        if not HAS_WEB:
            logger.warning("aiohttp not installed, REST API disabled")
            return
        
        self.app = web.Application()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        # Routes
        cors.add(self.app.router.add_get('/status', self.handle_status))
        cors.add(self.app.router.add_post('/organize', self.handle_organize))
        cors.add(self.app.router.add_get('/stats', self.handle_stats))
        cors.add(self.app.router.add_post('/undo', self.handle_undo))
        cors.add(self.app.router.add_post('/search', self.handle_search))
        cors.add(self.app.router.add_get('/preview/{file_id}', self.handle_preview))
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, 'localhost', self.port)
        await site.start()
        
        logger.info(f"🌐 REST API started on http://localhost:{self.port}")
    
    async def handle_status(self, request):
        """معالج API: الحصول على الحالة."""
        return web.json_response({
            'status': 'running',
            'stats': self.organizer.stats
        })
    
    async def handle_organize(self, request):
        """معالج API: تشغيل التنظيم."""
        try:
            data = await request.json()
            target = data.get('target_folder', self.organizer.config.target_folder)
            
            self.organizer.config.target_folder = target
            await self.organizer.run()
            
            return web.json_response({
                'success': True,
                'stats': self.organizer.stats
            })
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def handle_stats(self, request):
        """معالج API: الحصول على الإحصائيات."""
        return web.json_response(self.organizer.stats)
    
    async def handle_undo(self, request):
        """معالج API: التراجع عن آخر عملية."""
        try:
            await self.organizer.undo_last()
            return web.json_response({'success': True})
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def handle_search(self, request):
        """معالج API: البحث في الملفات."""
        try:
            data = await request.json()
            query = data.get('query', '')
            
            if not query:
                return web.json_response({
                    'success': False,
                    'error': 'Query parameter is required'
                }, status=400)
            
            results = self.organizer.db.search_content(query)
            return web.json_response({
                'success': True,
                'results': results
            })
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def handle_preview(self, request):
        """معالج API: الحصول على صورة معاينة للملف."""
        try:
            file_id = request.match_info['file_id']
            
            # Get file info from database
            cursor = self.organizer.db.conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,)
            )
            file_info = cursor.fetchone()
            
            if not file_info:
                return web.Response(status=404)
            
            # Check if preview exists
            preview_path = file_info.get('preview_path', '')
            if not preview_path or not Path(preview_path).exists():
                return web.Response(status=404)
            
            # Return the preview image
            with open(preview_path, 'rb') as f:
                return web.Response(body=f.read(), content_type='image/jpeg')
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def stop(self):
        """إيقاف خادم REST API."""
        if self.runner:
            await self.runner.cleanup()

# ==========================================
# ORGANIZER ENGINE
# ==========================================

class SmartOrganizer:
    """
    المحرك الأساسي الذي ينسق عملية تنظيم الملفات بأكملها.
    يربط بين المحلل، والمصنف، وكاشف التكرارات، وقاعدة البيانات.
    """
    def __init__(self, config: Config):
        """
        تهيئة المنظم الذكي.

        المعاملات:
            config (Config): كائن التكوين.
        """
        self.config = config
        self.target_dir = Path(config.target_folder)
        self.organized_root = Path(config.organized_root)
        self.backup_folder = Path(config.backup_folder)
        self.db = Database(config.cache_db)
        self.analyzer = FileAnalyzer(config)
        self.categorizer = None
        self.ml_clustering = MLClustering(config)
        self.duplicate_detector = AdvancedDuplicateDetector(config, self.db)
        self.cloud_manager = CloudBackupManager(config)
        self.rest_api = None
        self.stats = {
            'files_scanned': 0,
            'files_organized': 0,
            'duplicates_found': 0,
            'duplicates_handled': 0,
            'cloud_backups': 0,
            'errors': 0
        }
        self.scheduler_thread = None
    
    async def initialize(self):
        """
        مهام التهيئة غير المتزامنة (تهيئة قاعدة البيانات، بدء API، بدء المجدول).
        """
        self.db.init()
        self.categorizer = AICategorizer(self.config, self.db)
        
        # Start REST API if enabled
        if self.config.enable_rest_api:
            self.rest_api = RestAPI(self, self.config.rest_api_port)
            await self.rest_api.start()
        
        # Start scheduler if enabled
        if self.config.enable_scheduling:
            self._setup_scheduler()
        
        logger.info("✅ Organizer initialized")
    
    def _setup_scheduler(self):
        """إعداد الجدولة في الخلفية للتشغيل التلقائي."""
        def run_job():
            asyncio.run(self.run())
        
        if self.config.schedule_interval == "hourly":
            schedule.every().hour.at(self.config.schedule_time.split(":")[1]).do(run_job)
        elif self.config.schedule_interval == "daily":
            schedule.every().day.at(self.config.schedule_time).do(run_job)
        elif self.config.schedule_interval == "weekly":
            schedule.every().week.at(self.config.schedule_time).do(run_job)
        
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"📅 Scheduler set up: {self.config.schedule_interval} at {self.config.schedule_time}")
    
    async def run(self):
        """
        ينفذ سير عمل التنظيم الرئيسي:
        مسح -> اكتشاف التكرارات -> التعامل مع التكرارات -> التنظيم -> التقرير.
        """
        logger.info("🎯 Starting organization...")
        
        files = await self._scan_directory()
        if not files:
            logger.info("No files to organize")
            return
        
        logger.info(f"📊 Found {len(files)} files")
        
        # Find duplicates first
        duplicate_groups = await self.duplicate_detector.find_duplicates(files)
        duplicate_actions = await self.duplicate_detector.handle_duplicates(duplicate_groups)
        
        # Filter out files that will be skipped due to duplication
        files_to_organize = []
        skip_hashes = {action['file']['hash'] for action in duplicate_actions if action['action'] == 'skip'}
        
        for file_info in files:
            if file_info['hash'] not in skip_hashes:
                files_to_organize.append(file_info)
        
        self.stats['duplicates_found'] = sum(len(group) for group in duplicate_groups)
        self.stats['duplicates_handled'] = len(duplicate_actions)
        
        # Handle duplicate actions
        for action in duplicate_actions:
            await self._handle_duplicate_action(action)
        
        # Organize remaining files
        await self._organize_files(files_to_organize)
        
        self._print_report()
    
    async def _scan_directory(self) -> List[Dict]:
        """
        مسح الدليل المستهدف للملفات وتحليلها.

        القيمة المرجعة:
            List[Dict]: قائمة بمعلومات الملفات المحللة.
        """
        files = []
        for entry in self.target_dir.rglob('*'):
            if entry.is_file() and not entry.name.startswith('.'):
                if entry.suffix.lower() in self.config.allowed_extensions:
                    files.append(entry)
        
        self.stats['files_scanned'] = len(files)
        
        tasks = [self.analyzer.analyze(fp) for fp in files]
        return await asyncio.gather(*tasks)
    
    async def _organize_files(self, files: List[Dict]):
        """
        تنظيم الملفات عن طريق التجميع ثم معالجة الدفعات.

        المعاملات:
            files (List[Dict]): قائمة بالملفات لتنظيمها.
        """
        # ML Clustering if enabled
        if self.config.enable_ml_clustering:
            clusters = self.ml_clustering.cluster_files(files)
            logger.info(f"📊 Processing {len(clusters)} ML clusters")
            
            for cluster_id, cluster_files in clusters.items():
                await self._process_batch(cluster_files)
        else:
            await self._process_batch(files)
    
    async def _process_batch(self, files: List[Dict]):
        """
        تصنيف ونقل دفعة من الملفات.

        المعاملات:
            files (List[Dict]): دفعة من الملفات.
        """
        batch_size = self.config.batch_size
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            categories = await self.categorizer.categorize_batch(batch)
            
            for file_info in batch:
                await self._move_file(file_info, categories.get(file_info['path']))
    
    async def _move_file(self, file_info: Dict, category_data: Optional[Dict]):
        """
        نقل الملف إلى مجلد الفئة المخصص له.

        المعاملات:
            file_info (Dict): البيانات الوصفية للملف.
            category_data (Optional[Dict]): نتيجة التصنيف.
        """
        if not category_data:
            return
        
        try:
            folder = category_data.get('folder', 'Others')
            new_name = category_data.get('new_name') if self.config.smart_rename else None
            
            dest_folder = self.organized_root / folder
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            dest_name = new_name or file_info['name']
            dest_path = dest_folder / dest_name
            
            # Handle existing files
            if dest_path.exists():
                if self.config.duplicate_strategy == DuplicateStrategy.VERSION.value:
                    # Create a versioned copy
                    version = 1
                    base, ext = os.path.splitext(dest_name)
                    
                    while dest_path.exists():
                        dest_path = dest_folder / f"{base}_v{version}{ext}"
                        version += 1
                    
                    # Log the version
                    if self.config.enable_versioning:
                        self.db.add_file_version(file_info['hash'], str(dest_path), version)
                
                elif self.config.duplicate_strategy == DuplicateStrategy.RENAME.value:
                    base, ext = os.path.splitext(dest_name)
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_folder / f"{base}_{counter}{ext}"
                        counter += 1
                
                elif self.config.duplicate_strategy == DuplicateStrategy.SKIP.value:
                    logger.info(f"⏭️ Skipping duplicate: {file_info['name']}")
                    return
            
            if not self.config.dry_run:
                shutil.move(file_info['path'], str(dest_path))
                self.db.log_operation('move', file_info['path'], str(dest_path))
                
                # Cloud backup if enabled
                if self.config.enable_cloud_backup:
                    cloud_url = await self.cloud_manager.backup_file(dest_path, folder)
                    if cloud_url:
                        self.stats['cloud_backups'] += 1
                        # Update database with cloud backup info
                        self.db.conn.execute(
                            "UPDATE files SET cloud_backup = 1 WHERE hash = ?",
                            (file_info['hash'],)
                        )
                        self.db.conn.commit()
                
                logger.info(f"✓ {file_info['name']} → {folder}/")
            else:
                logger.info(f"[DRY RUN] {file_info['name']} → {folder}/")
            
            self.stats['files_organized'] += 1
        
        except Exception as e:
            logger.error(f"❌ Failed to move {file_info['name']}: {e}")
            self.stats['errors'] += 1
    
    async def _handle_duplicate_action(self, action: Dict):
        """
        تنفيذ إجراء محدد لملف مكرر.

        المعاملات:
            action (Dict): تفاصيل الإجراء (النوع، معلومات الملف).
        """
        file_info = action['file']
        action_type = action['action']
        
        if action_type == 'skip':
            logger.info(f"⏭️ Skipping duplicate: {file_info['name']}")
            return
        
        elif action_type == 'version':
            # This is handled in _move_file
            return
        
        elif action_type == 'move_to_backup':
            try:
                backup_path = self.backup_folder / file_info['name']
                shutil.move(file_info['path'], str(backup_path))
                logger.info(f"📦 Moved duplicate to backup: {file_info['name']}")
            except Exception as e:
                logger.error(f"❌ Failed to move duplicate to backup: {e}")
                self.stats['errors'] += 1
    
    def _print_report(self):
        """طباعة تقرير ملخص لجلسة التنظيم."""
        print("\n" + "=" * 60)
        print("📊 ORGANIZATION REPORT")
        print("=" * 60)
        print(f"📁 Files scanned: {self.stats['files_scanned']}")
        print(f"✅ Files organized: {self.stats['files_organized']}")
        print(f"🔍 Duplicates found: {self.stats['duplicates_found']}")
        print(f"🔧 Duplicates handled: {self.stats['duplicates_handled']}")
        if self.config.enable_cloud_backup:
            print(f"☁️ Cloud backups: {self.stats['cloud_backups']}")
        print(f"❌ Errors: {self.stats['errors']}")
        print("=" * 60 + "\n")
    
    async def undo_last(self):
        """التراجع عن أحدث عملية (إعادة نقل الملف إلى موقعه الأصلي)."""
        ops = self.db.get_recent_operations(limit=1)
        if ops:
            op = ops[0]
            try:
                shutil.move(op['dest_path'], op['file_path'])
                logger.info(f"↩️ Undone: {Path(op['file_path']).name}")
            except Exception as e:
                logger.error(f"Undo failed: {e}")
    
    async def search_files(self, query: str) -> List[Dict]:
        """
        البحث عن الملفات باستخدام فهرس قاعدة البيانات.

        المعاملات:
            query (str): نص البحث.

        القيمة المرجعة:
            List[Dict]: نتائج البحث.
        """
        return self.db.search_content(query)
    
    async def shutdown_async(self):
        """تنفيذ مهام الإيقاف غير المتزامنة."""
        if self.rest_api:
            await self.rest_api.stop()
        self.db.close()
        logger.info("👋 Shutdown complete")
    
    def shutdown(self):
        """غلاف متزامن لعملية الإيقاف."""
        asyncio.run(self.shutdown_async())

# ==========================================
# GUI
# ==========================================

class OrganizerGUI:
    """
    واجهة مستخدم رسومية تعتمد على Tkinter للمنظم.
    """
    def __init__(self, root: tk.Tk, config: Config):
        """
        تهيئة واجهة المستخدم الرسومية.

        المعاملات:
            root (tk.Tk): نافذة Tkinter الجذرية.
            config (Config): كائن التكوين.
        """
        self.root = root
        self.root.title("Smart Organizer Ultimate - Enhanced Edition")
        self.root.geometry("1000x700")
        
        self.config = config
        self.organizer = SmartOrganizer(config)
        self.is_running = False
        self.current_preview = None
        
        self._setup_ui()
        
        threading.Thread(target=self._init_async, daemon=True).start()
    
    def _init_async(self):
        """تهيئة محرك المنظم في خيط (Thread) في الخلفية."""
        asyncio.run(self.organizer.initialize())
    
    def _setup_ui(self):
        """إعداد مكونات واجهة المستخدم الرئيسية (علامات التبويب)."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Organization")
        
        # Search tab
        search_frame = ttk.Frame(notebook)
        notebook.add(search_frame, text="Search")
        
        # Settings tab
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        
        # Setup main tab
        self._setup_main_tab(main_frame)
        
        # Setup search tab
        self._setup_search_tab(search_frame)
        
        # Setup settings tab
        self._setup_settings_tab(settings_frame)
    
    def _setup_main_tab(self, parent):
        """إعداد علامة تبويب 'التنظيم'."""
        # Header
        header = ttk.Label(parent, text="Smart File Organizer Ultimate - Enhanced", 
                          font=("Arial", 16, "bold"))
        header.pack(pady=20)
        
        # Folder selection
        folder_frame = ttk.Frame(parent)
        folder_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(folder_frame, text="Target Folder:").pack(side=tk.LEFT)
        self.folder_var = tk.StringVar(value=self.config.target_folder)
        ttk.Entry(folder_frame, textvariable=self.folder_var, width=50).pack(side=tk.LEFT, padx=10)
        ttk.Button(folder_frame, text="Browse", command=self._browse).pack(side=tk.LEFT)
        
        # Options
        options_frame = ttk.LabelFrame(parent, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.dry_run_var = tk.BooleanVar(value=self.config.dry_run)
        ttk.Checkbutton(options_frame, text="Dry Run (No Changes)", 
                       variable=self.dry_run_var).pack(anchor=tk.W)
        
        self.smart_rename_var = tk.BooleanVar(value=self.config.smart_rename)
        ttk.Checkbutton(options_frame, text="Smart Rename", 
                       variable=self.smart_rename_var).pack(anchor=tk.W)
        
        self.ml_clustering_var = tk.BooleanVar(value=self.config.enable_ml_clustering)
        ttk.Checkbutton(options_frame, text="ML Clustering (Group Similar Files)", 
                       variable=self.ml_clustering_var).pack(anchor=tk.W)
        
        self.content_analysis_var = tk.BooleanVar(value=self.config.enable_content_analysis)
        ttk.Checkbutton(options_frame, text="Content Analysis", 
                       variable=self.content_analysis_var).pack(anchor=tk.W)
        
        self.versioning_var = tk.BooleanVar(value=self.config.enable_versioning)
        ttk.Checkbutton(options_frame, text="File Versioning", 
                       variable=self.versioning_var).pack(anchor=tk.W)
        
        self.cloud_backup_var = tk.BooleanVar(value=self.config.enable_cloud_backup)
        ttk.Checkbutton(options_frame, text="Cloud Backup", 
                       variable=self.cloud_backup_var).pack(anchor=tk.W)
        
        self.rest_api_var = tk.BooleanVar(value=self.config.enable_rest_api)
        ttk.Checkbutton(options_frame, text="Enable REST API", 
                       variable=self.rest_api_var).pack(anchor=tk.W)
        
        # Duplicate strategy
        dup_frame = ttk.Frame(options_frame)
        dup_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dup_frame, text="Duplicate Strategy:").pack(side=tk.LEFT)
        self.dup_strategy_var = tk.StringVar(value=self.config.duplicate_strategy)
        dup_combo = ttk.Combobox(dup_frame, textvariable=self.dup_strategy_var, 
                                 values=[s.value for s in DuplicateStrategy], width=15)
        dup_combo.pack(side=tk.LEFT, padx=10)
        
        # Progress
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=20, pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Organize Now", 
                  command=self._start_organize).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Undo Last", 
                  command=self._undo).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Open Organized Folder", 
                  command=self._open_organized_folder).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var).pack(side=tk.BOTTOM, pady=10)
    
    def _setup_search_tab(self, parent):
        """إعداد علامة تبويب 'البحث'."""
        # Search frame
        search_frame = ttk.Frame(parent)
        search_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=50)
        search_entry.pack(side=tk.LEFT, padx=10)
        ttk.Button(search_frame, text="Search", command=self._search).pack(side=tk.LEFT)
        
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Search Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Treeview for results
        columns = ('name', 'path', 'category', 'modified')
        self.search_tree = ttk.Treeview(results_frame, columns=columns, show='headings')
        
        for col in columns:
            self.search_tree.heading(col, text=col.capitalize())
            self.search_tree.column(col, width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.search_tree.yview)
        self.search_tree.configure(yscrollcommand=scrollbar.set)
        
        self.search_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.preview_canvas = Canvas(preview_frame, bg='white', height=200)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event
        self.search_tree.bind('<<TreeviewSelect>>', self._on_search_select)
    
    def _setup_settings_tab(self, parent):
        """إعداد علامة تبويب 'الإعدادات'."""
        # Create notebook for sub-settings
        settings_notebook = ttk.Notebook(parent)
        settings_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # General settings
        general_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(general_frame, text="General")
        
        # API settings
        api_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(api_frame, text="API")
        
        # Cloud settings
        cloud_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(cloud_frame, text="Cloud")
        
        # Schedule settings
        schedule_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(schedule_frame, text="Schedule")
        
        # Setup each settings tab
        self._setup_general_settings(general_frame)
        self._setup_api_settings(api_frame)
        self._setup_cloud_settings(cloud_frame)
        self._setup_schedule_settings(schedule_frame)
    
    def _setup_general_settings(self, parent):
        """الإعدادات الفرعية: التكوين العام."""
        # Batch size
        batch_frame = ttk.Frame(parent)
        batch_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.IntVar(value=self.config.batch_size)
        ttk.Spinbox(batch_frame, from_=1, to=100, textvariable=self.batch_size_var, 
                   width=10).pack(side=tk.LEFT, padx=10)
        
        # Max workers
        workers_frame = ttk.Frame(parent)
        workers_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(workers_frame, text="Max Workers:").pack(side=tk.LEFT)
        self.max_workers_var = tk.IntVar(value=self.config.max_workers)
        ttk.Spinbox(workers_frame, from_=1, to=20, textvariable=self.max_workers_var, 
                   width=10).pack(side=tk.LEFT, padx=10)
        
        # Custom categories
        cat_frame = ttk.LabelFrame(parent, text="Custom Categories", padding=10)
        cat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # This would need a more complex UI for editing categories
        # For simplicity, just showing the current categories
        cat_text = tk.Text(cat_frame, height=10)
        cat_text.pack(fill=tk.BOTH, expand=True)
        
        cat_text.insert(tk.END, json.dumps(self.config.custom_categories, indent=2))
        cat_text.config(state=tk.DISABLED)
        
        # Save button
        ttk.Button(parent, text="Save Settings", command=self._save_settings).pack(pady=20)
    
    def _setup_api_settings(self, parent):
        """الإعدادات الفرعية: مفاتيح API والمنافذ."""
        # API Key
        api_frame = ttk.Frame(parent)
        api_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(api_frame, text="Gemini API Key:").pack(anchor=tk.W)
        self.api_key_var = tk.StringVar(value=self.config.api_key)
        api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*")
        api_entry.pack(fill=tk.X, pady=5)
        
        # REST API Port
        port_frame = ttk.Frame(parent)
        port_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(port_frame, text="REST API Port:").pack(side=tk.LEFT)
        self.port_var = tk.IntVar(value=self.config.rest_api_port)
        ttk.Spinbox(port_frame, from_=1000, to=9999, textvariable=self.port_var, 
                   width=10).pack(side=tk.LEFT, padx=10)
        
        # Save button
        ttk.Button(parent, text=" Save API Settings", command=self._save_settings).pack(pady=20)
    
    def _setup_cloud_settings(self, parent):
        """الإعدادات الفرعية: تكوين مزود السحابة."""
        # Cloud provider
        provider_frame = ttk.Frame(parent)
        provider_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(provider_frame, text="Cloud Provider:").pack(side=tk.LEFT)
        self.cloud_provider_var = tk.StringVar(value=self.config.cloud_provider)
        provider_combo = ttk.Combobox(provider_frame, textvariable=self.cloud_provider_var, 
                                     values=[p.value for p in CloudProvider], width=15)
        provider_combo.pack(side=tk.LEFT, padx=10)
        
        # AWS S3 settings
        s3_frame = ttk.LabelFrame(parent, text="AWS S3 Settings", padding=10)
        s3_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(s3_frame, text="Access Key:").grid(row=0, column=0, sticky=tk.W)
        self.aws_access_key_var = tk.StringVar(value=self.config.cloud_settings.get('aws_access_key', ''))
        ttk.Entry(s3_frame, textvariable=self.aws_access_key_var, width=30, show="*").grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(s3_frame, text="Secret Key:").grid(row=1, column=0, sticky=tk.W)
        self.aws_secret_key_var = tk.StringVar(value=self.config.cloud_settings.get('aws_secret_key', ''))
        ttk.Entry(s3_frame, textvariable=self.aws_secret_key_var, width=30, show="*").grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(s3_frame, text="Region:").grid(row=2, column=0, sticky=tk.W)
        self.aws_region_var = tk.StringVar(value=self.config.cloud_settings.get('aws_region', 'us-east-1'))
        ttk.Entry(s3_frame, textvariable=self.aws_region_var, width=30).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(s3_frame, text="Bucket Name:").grid(row=3, column=0, sticky=tk.W)
        self.aws_bucket_var = tk.StringVar(value=self.config.cloud_settings.get('s3_bucket', ''))
        ttk.Entry(s3_frame, textvariable=self.aws_bucket_var, width=30).grid(row=3, column=1, padx=5, pady=5)
        
        # Save button
        ttk.Button(parent, text="Save Cloud Settings", command=self._save_settings).pack(pady=20)
    
    def _setup_schedule_settings(self, parent):
        """الإعدادات الفرعية: تكوين الجدولة."""
        # Enable scheduling
        enable_frame = ttk.Frame(parent)
        enable_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.enable_schedule_var = tk.BooleanVar(value=self.config.enable_scheduling)
        ttk.Checkbutton(enable_frame, text="Enable Scheduled Organization", 
                       variable=self.enable_schedule_var).pack(anchor=tk.W)
        
        # Schedule interval
        interval_frame = ttk.Frame(parent)
        interval_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(interval_frame, text="Schedule Interval:").pack(side=tk.LEFT)
        self.schedule_interval_var = tk.StringVar(value=self.config.schedule_interval)
        interval_combo = ttk.Combobox(interval_frame, textvariable=self.schedule_interval_var, 
                                     values=["hourly", "daily", "weekly"], width=15)
        interval_combo.pack(side=tk.LEFT, padx=10)
        
        # Schedule time
        time_frame = ttk.Frame(parent)
        time_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(time_frame, text="Time (HH:MM):").pack(side=tk.LEFT)
        self.schedule_time_var = tk.StringVar(value=self.config.schedule_time)
        ttk.Entry(time_frame, textvariable=self.schedule_time_var, width=10).pack(side=tk.LEFT, padx=10)
        
        # Save button
        ttk.Button(parent, text="Save Schedule Settings", command=self._save_settings).pack(pady=20)
    
    def _browse(self):
        """فتح مربع حوار استعراض الدليل."""
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)
    
    def _start_organize(self):
        """التحقق من صحة الإعدادات وبدء عملية التنظيم."""
        if self.is_running:
            return
        
        self.config.target_folder = self.folder_var.get()
        self.config.dry_run = self.dry_run_var.get()
        self.config.smart_rename = self.smart_rename_var.get()
        self.config.enable_ml_clustering = self.ml_clustering_var.get()
        self.config.enable_content_analysis = self.content_analysis_var.get()
        self.config.enable_versioning = self.versioning_var.get()
        self.config.enable_cloud_backup = self.cloud_backup_var.get()
        self.config.enable_rest_api = self.rest_api_var.get()
        self.config.duplicate_strategy = self.dup_strategy_var.get()
        
        self.is_running = True
        self.status_var.set("Organizing...")
        self.progress.start()
        
        threading.Thread(target=self._run_organize, daemon=True).start()
    
    def _run_organize(self):
        """عامل الخيط لتشغيل التنظيم."""
        try:
            asyncio.run(self.organizer.run())
            self.root.after(0, self._on_complete)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, self._on_complete)
    
    def _on_complete(self):
        """رد الاتصال عند اكتمال التنظيم."""
        self.is_running = False
        self.progress.stop()
        self.status_var.set("Ready")
        messagebox.showinfo("Complete", "Organization completed!")
    
    def _undo(self):
        """تشغيل عملية التراجع."""
        asyncio.run(self.organizer.undo_last())
    
    def _open_organized_folder(self):
        """فتح المجلد المنظم في مستكشف ملفات النظام."""
        path = self.config.organized_root
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    
    def _search(self):
        """بدء البحث عن الملفات."""
        query = self.search_var.get().strip()
        if not query:
            return
        
        # Clear current results
        for item in self.search_tree.get_children():
            self.search_tree.delete(item)
        
        # Search in background thread
        threading.Thread(target=self._run_search, args=(query,), daemon=True).start()
    
    def _run_search(self, query):
        """عامل الخيط للبحث."""
        try:
            results = asyncio.run(self.organizer.search_files(query))
            
            # Update UI in main thread
            self.root.after(0, lambda: self._update_search_results(results))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Search Error", str(e)))
    
    def _update_search_results(self, results):
        """ملء شجرة نتائج البحث."""
        for result in results:
            self.search_tree.insert('', tk.END, values=(
                result['name'],
                result['path'],
                result['category'],
                result['modified']
            ))
    
    def _on_search_select(self, event):
        """التعامل مع اختيار عنصر نتيجة البحث."""
        selection = self.search_tree.selection()
        if not selection:
            return
        
        item = self.search_tree.item(selection[0])
        file_path = item['values'][1]
        
        # Clear previous preview
        self.preview_canvas.delete("all")
        
        # Try to show preview
        threading.Thread(target=self._load_preview, args=(file_path,), daemon=True).start()
    
    def _load_preview(self, file_path):
        """تحميل معاينة الملف (صورة أو نص) في الخلفية."""
        try:
            # Check if it's an image
            ext = Path(file_path).suffix.lower()
            if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}:
                # Load and resize image
                img = Image.open(file_path)
                img.thumbnail((300, 300))
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Update canvas in main thread
                self.root.after(0, lambda: self._show_image_preview(photo))
            
            # For text files, show content preview
            elif ext in {'.txt', '.md', '.py', '.js', '.html', '.css', '.json'}:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1000)  # First 1000 chars
                
                # Update canvas in main thread
                self.root.after(0, lambda: self._show_text_preview(content))
        
        except Exception as e:
            logger.error(f"Failed to load preview: {e}")
    
    def _show_image_preview(self, photo):
        """عرض معاينة الصورة على القماش (Canvas)."""
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(150, 100, image=photo)
        self.current_preview = photo  # Keep reference
    
    def _show_text_preview(self, content):
        """عرض معاينة النص على القماش (Canvas)."""
        self.preview_canvas.delete("all")
        # Simple text display (would need more sophisticated implementation for better results)
        lines = content.split('\n')[:20]  # First 20 lines
        for i, line in enumerate(lines):
            self.preview_canvas.create_text(10, 20 + i*15, text=line, anchor=tk.W)
    
    def _save_settings(self):
        """حفظ إعدادات واجهة المستخدم الحالية في التكوين."""
        # Update config with values from UI
        self.config.batch_size = self.batch_size_var.get()
        self.config.max_workers = self.max_workers_var.get()
        self.config.api_key = self.api_key_var.get()
        self.config.rest_api_port = self.port_var.get()
        self.config.cloud_provider = self.cloud_provider_var.get()
        self.config.enable_scheduling = self.enable_schedule_var.get()
        self.config.schedule_interval = self.schedule_interval_var.get()
        self.config.schedule_time = self.schedule_time_var.get()
        
        # Update cloud settings
        self.config.cloud_settings = {
            'aws_access_key': self.aws_access_key_var.get(),
            'aws_secret_key': self.aws_secret_key_var.get(),
            'aws_region': self.aws_region_var.get(),
            's3_bucket': self.aws_bucket_var.get()
        }
        
        # Save config
        self.config.save()
        messagebox.showinfo("Success", "Settings saved!")

# ==========================================
# MAIN
# ==========================================

async def main_cli():
    """
    نقطة الدخول الرئيسية لتحليل وسائط سطر الأوامر (CLI) وبدء تشغيل التطبيق.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Organizer Ultimate - Enhanced")
    parser.add_argument('--target', '-t', help='Target folder')
    parser.add_argument('--api-key', '-k', help='Gemini API key')
    parser.add_argument('--dry-run', action='store_true', help='Simulate only')
    parser.add_argument('--no-gui', action='store_true', help='CLI mode')
    parser.add_argument('--enable-api', action='store_true', help='Enable REST API')
    parser.add_argument('--api-port', type=int, default=8080, help='REST API port')
    parser.add_argument('--enable-ml', action='store_true', help='Enable ML clustering')
    parser.add_argument('--enable-content-analysis', action='store_true', help='Enable content analysis')
    parser.add_argument('--enable-versioning', action='store_true', help='Enable file versioning')
    parser.add_argument('--enable-cloud-backup', action='store_true', help='Enable cloud backup')
    parser.add_argument('--cloud-provider', choices=[p.value for p in CloudProvider], default='none', help='Cloud provider')
    parser.add_argument('--enable-scheduling', action='store_true', help='Enable scheduled organization')
    parser.add_argument('--schedule-interval', choices=['hourly', 'daily', 'weekly'], default='daily', help='Schedule interval')
    parser.add_argument('--schedule-time', default='02:00', help='Schedule time (HH:MM)')
    parser.add_argument('--undo', action='store_true', help='Undo last operation')
    parser.add_argument('--search', help='Search for files')
    
    args = parser.parse_args()
    
    config = Config.load()
    
    if args.target:
        config.target_folder = args.target
        config.__post_init__()
    
    if args.api_key:
        config.api_key = args.api_key
    
    if args.dry_run:
        config.dry_run = True
    
    if args.enable_api:
        config.enable_rest_api = True
        config.rest_api_port = args.api_port
    
    if args.enable_ml:
        config.enable_ml_clustering = True
    
    if args.enable_content_analysis:
        config.enable_content_analysis = True
    
    if args.enable_versioning:
        config.enable_versioning = True
    
    if args.enable_cloud_backup:
        config.enable_cloud_backup = True
        config.cloud_provider = args.cloud_provider
    
    if args.enable_scheduling:
        config.enable_scheduling = True
        config.schedule_interval = args.schedule_interval
        config.schedule_time = args.schedule_time
    
    global logger
    logger = setup_logging(config)
    
    if args.no_gui or not config.enable_gui:
        organizer = SmartOrganizer(config)
        await organizer.initialize()
        
        if args.undo:
            await organizer.undo_last()
        elif args.search:
            results = await organizer.search_files(args.search)
            for result in results:
                print(f"{result['name']} ({result['category']}) - {result['path']}")
        else:
            await organizer.run()
            
            # Keep running if REST API enabled
            if config.enable_rest_api:
                logger.info("🌐 REST API running. Press Ctrl+C to stop")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    pass
        
        await organizer.shutdown_async()
    else:
        root = tk.Tk()
        app = OrganizerGUI(root, config)
        root.mainloop()

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════╗
║     Smart Downloads Organizer Ultimate - Enhanced      ║
║     Advanced File Organization with AI & ML             ║
╚════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main_cli())
