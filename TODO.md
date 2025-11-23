# TODO - المهام المخططة

## مشروع جديد: نظام إدارة وجبات الطاقم (CrewCatering)

### Master Prompt: Full Stack Crew Catering System (Arabic-First)

#### الدور والهدف
أنت مهندس معماري رئيسي للبرمجيات (Full-Stack). مهمتك هي بناء نظام جاهز للإنتاج يسمى **"CrewCatering"**.
**القيد الأساسي:** هذا تطبيق **عربي أولاً (Arabic-First)**. تخطيط واجهة المستخدم الافتراضي هو **RTL (من اليمين إلى اليسار)**.

#### الحزمة التقنية والمعايير

1. **الواجهة الأمامية (Frontend):** Angular 18+ (Standalone Components)، NgRx Signals (إدارة الحالة)، Angular Material
2. **الواجهة الخلفية (Backend):** Python FastAPI (غير متزامن)، SQLAlchemy (غير متزامن)، PostgreSQL + PostGIS
3. **المعمارية:** التصميم الموجه بالنطاق (DDD) على كلا الطرفين
4. **اللغة:** العربية (افتراضية) + الإنجليزية (ثانوية)

---

### الجزء 1: الواجهة الخلفية (Backend - FastAPI)

إنشاء هيكل API قابل للتوسع يلبي المتطلبات التالية:

#### قاعدة البيانات
- استخدام محرك `AsyncPG`
- يجب أن تدعم النماذج UTF-8 بالكامل
- استخدام `geoalchemy2` لتكامل PostGIS
- تنفيذ استعلام للعثور على المطاعم ضمن نطاق 3 كم

#### المصادقة (Auth)
- مبنية على JWT
- نقطة نهاية `POST /auth/qr-login` تأخذ حمولة موقعة

#### محرك التمويل (Finance Engine - العقل المدبر)
منطق الأعمال:
- **VIP**: مغطى بنسبة 100%
- **Standard**: المستخدم يدفع، إلا إذا كان هناك **استثناء نشط** (مرة كل 3 أسابيع)

#### دعم التدويل (i18n)
- ضمان إرجاع رسائل خطأ API بناءً على رأس `Accept-Language`

---

### الجزء 2: الواجهة الأمامية (Frontend - Angular 18+)

#### معمارية RTL (حرجة - Critical)
**ملاحظة مهمة:** هذه النقطة أساسية لتجنب كابوس التحويل لاحقاً

- استخدام **الخصائص المنطقية (Logical Properties)** في SCSS
  - مثال: `margin-inline-start` بدلاً من `margin-left`
  - مثال: `padding-inline-end` بدلاً من `padding-right`
  - هذا يضمن التبديل التلقائي للإنجليزية
- تعيين `dir="rtl"` في `index.html` افتراضياً

#### التدويل (Internationalization)
- استخدام **Transloco** أو **ngx-translate** للتبديل اللغوي في وقت التشغيل
  - مطلوب لميزة "التبديل الفوري" بدون إعادة تحميل التطبيق
- إنشاء ملفين JSON:
  - `ar.json` (أساسي)
  - `en.json` (ثانوي)

#### إدارة الحالة (State Management)
- تخزين `UserPreference { language: 'ar' | 'en' }` في LocalStorage
- المزامنة مع Signal

#### المكونات الرئيسية

##### 1. LanguageSwitcherComponent
- يبدل سمة `dir` على عنصر `html`
- يبدل ملفات الترجمة

##### 2. TrackingComponent
- يدمج الخريطة (Leaflet أو Google Maps)
- **حرج:** ضمان وضع عناصر التحكم في الخريطة بشكل صحيح في وضع RTL

---

### خطة التنفيذ

توفير المخرجات في 3 كتل متميزة:

#### الكتلة 1: قاعدة البيانات وجوهر الواجهة الخلفية
- `models.py` (SQLAlchemy مع أنواع جغرافية)
- `finance_service.py` (منطق تقسيم التكلفة)

#### الكتلة 2: معمارية الواجهة الأمامية وإعداد RTL
- `styles.scss` (تعريف mixins عالمية لـ RTL/LTR والخصائص المنطقية)
- `transloco.config.ts` (إعداد العربية كافتراضي)
- `app.component.ts` (منطق التعامل مع التبديل الديناميكي لـ `dir`)

#### الكتلة 3: تطبيق الميزات الرئيسية
- `GeoService` (الواجهة الخلفية) + `TrackingComponent` (الواجهة الأمامية)

---

### نصائح تقنية مهمة

#### 1. CSS Logical Properties (أهم نقطة في RTL)
- ❌ **خطأ:** استخدام `margin-left`
- ✅ **صحيح:** استخدام `margin-inline-start`
- هذه الطريقة الحديثة تجعل CSS يقلب تلقائياً عند تغيير اللغة

#### 2. الخطوط (Fonts)
في ملف SCSS:
- **للعربية:** استخدام خطوط "Tajawal" أو "Cairo" من Google Fonts
- **للإنجليزية:** استخدام "Roboto"
- هذا يحدث فرقاً كبيراً في المظهر الاحترافي ("رصين وجاد")

#### 3. خرائط RTL
- أزرار التحكم في الخريطة (الزووم وغيره) قد تتعارض مع RTL
- التأكد من أن CSS الخاص بالخريطة معزول (Isolated)
- أو تطبيق `dir="ltr"` إذا كانت الخريطة بالإنجليزية
- أو ضبطها بشكل صحيح للعربية

---

### الأولويات والمراحل

#### المرحلة 1: البنية التحتية (Infrastructure)
- [ ] إعداد مشروع Angular 18+ بوضع Standalone
- [ ] إعداد مشروع FastAPI مع هيكل DDD
- [ ] تكوين PostgreSQL + PostGIS
- [ ] إعداد نظام المصادقة JWT

#### المرحلة 2: الميزات الأساسية (Core Features)
- [ ] نظام تسجيل الدخول عبر QR
- [ ] محرك التمويل (Finance Engine)
- [ ] خدمة الموقع الجغرافي (GeoService)
- [ ] نظام إدارة المستخدمين (VIP/Standard)

#### المرحلة 3: واجهة المستخدم (UI/UX)
- [ ] تنفيذ نظام RTL الكامل
- [ ] مبدل اللغة (Language Switcher)
- [ ] لوحة القيادة (Dashboard)
- [ ] مكون التتبع مع الخريطة

#### المرحلة 4: التحسينات والاختبارات
- [ ] اختبارات الوحدة (Unit Tests)
- [ ] اختبارات التكامل (Integration Tests)
- [ ] اختبارات RTL/LTR
- [ ] تحسين الأداء

---

### ملاحظات إضافية

#### الهوية البصرية (Visual Identity)
- التطبيق يجب أن يكون "رصين وجاد"
- استخدام ألوان احترافية
- تجربة مستخدم سلسة وبسيطة

#### الأمان (Security)
- التحقق من صحة جميع المدخلات
- حماية ضد OWASP Top 10
- تشفير البيانات الحساسة

#### الأداء (Performance)
- استخدام Lazy Loading للمكونات
- تحسين استعلامات قاعدة البيانات
- التخزين المؤقت (Caching) حيثما أمكن

---

### المراجع والموارد

#### للواجهة الأمامية
- [Angular 18 Documentation](https://angular.dev)
- [Angular Material RTL Guide](https://material.angular.io/guide/typography)
- [Transloco Documentation](https://ngneat.github.io/transloco/)

#### للواجهة الخلفية
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [PostGIS Documentation](https://postgis.net/documentation/)

---

### الخلاصة

هذا المشروع يمثل **"وثيقة معمارية" (Architectural Blueprint)** متكاملة تجمع:

1. **المتانة:** FastAPI + Async + PostGIS
2. **الحداثة:** Angular 18 + Signals + Standalone
3. **الواقعية:** قواعد VIP والاستثناءات
4. **الهوية:** نظام عربي أولاً (Arabic-First)

**الإضافة الأهم:** النظام **"عربي أولاً"** وليس مجرد ترجمة، مما يتطلب بناءً صحيحاً من "الدقيقة صفر" لتجنب كابوس التحويل لاحقاً.
