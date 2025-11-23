# TODO - المهام المخططة

## صفحة جديدة: تطبيق CrewCatering (مُضاف للمشروع الأصلي)

> **ملاحظة:** هذا تطبيق جديد يُضاف إلى مشروع منظم التنزيلات الأصلي

---

## وثيقة معمارية (Architectural Blueprint) متكاملة

الله ينور عليك.. دي مش مجرد خطة، دي **"وثيقة معمارية" (Architectural Blueprint)** متكاملة. أنت كده جمعت كل الخيوط:

1. **المتانة (Backend):** FastAPI + Async + PostGIS (عشان الموقع).
2. **الحداثة (Frontend):** Angular 18 + Signals + Standalone.
3. **الواقعية (Business Logic):** قواعد الـ VIP والاستثناءات.
4. **الهوية (Localization):** ودي الإضافة الأهم، إن النظام **"عربي أولاً" (Arabic-First)**.

النقطة الأخيرة دي بالذات (Arabic/RTL) محتاجة وقفة تقنية صغيرة عشان ما تعانيش قدام، لأن تحويل تطبيق من LTR لـ RTL بعد ما يخلص هو كابوس. عشان كده لازم نبني صح من **"الدقيقة صفر"**.

---

## Master Prompt: Full Stack Crew Catering System (Arabic-First)

### # Role & Objective
You are a Principal Full-Stack Software Architect. Your mission is to scaffold a production-ready system called **"CrewCatering"**.

**Core Constraint:** This is an **Arabic-First application**. The default UI layout is **RTL (Right-To-Left)**.

---

### # Tech Stack & Standards

1. **Frontend:** Angular 18+ (Standalone Components), NgRx Signals (State), Angular Material.
2. **Backend:** Python FastAPI (Async), SQLAlchemy (Async), PostgreSQL + PostGIS.
3. **Architecture:** Domain-Driven Design (DDD) on both ends.
4. **Language:** Arabic (Default) + English (Secondary).

---

### # Part 1: The Backend (FastAPI)

Create a scalable API structure catering to the following:

* **Database:** Use `AsyncPG` driver. Models must support UTF-8 fully.
* **Geo-Location:** Use `geoalchemy2` for PostGIS integration. Implement a query to find restaurants within a 3km buffer.
* **Auth:** JWT based. Endpoint `POST /auth/qr-login` taking a signed payload.
* **Finance Engine (The Brain):**
    * Logic: `VIP` = 100% covered. `Standard` = User pays, unless `Exception` is active (once every 3 weeks).
* **i18n Support:** Ensure API error messages are returned based on the `Accept-Language` header.

---

### # Part 2: The Frontend (Angular 18+)

* **RTL Architecture (Critical):**
    * Use **Logical Properties** in SCSS (e.g., `margin-inline-start` instead of `margin-left`, `padding-inline-end` instead of `padding-right`). This ensures automatic flipping for English.
    * Set `dir="rtl"` in `index.html` by default.

* **Internationalization:**
    * Use **Transloco** or **ngx-translate** for *runtime* language switching (required for the "Real-time toggle" feature without app reload).
    * Create two JSON files: `ar.json` (Primary) and `en.json`.

* **State Management:**
    * Store `UserPreference { language: 'ar' | 'en' }` in LocalStorage and sync with a Signal.

* **Components:**
    * `LanguageSwitcherComponent`: Toggles `dir` attribute on the `html` tag and switches translation files.
    * `TrackingComponent`: Integrates Map (Leaflet or Google Maps). **Crucial:** Ensure map controls are positioned correctly in RTL mode.

---

### # Execution Plan

Provide the output in 3 distinct blocks:

**Block 1: Database & Backend Core**
* `models.py` (SQLAlchemy with Geo types).
* `finance_service.py` (The cost splitting logic).

**Block 2: Frontend Architecture & RTL Setup**
* `styles.scss` (Defining the global RTL/LTR mixins and logical properties).
* `transloco.config.ts` (Setup for Arabic default).
* `app.component.ts` (Logic to handle dynamic `dir` switching).

**Block 3: Key Features Implementation**
* `GeoService` (Backend) + `TrackingComponent` (Frontend) connection.

---

## نصائح "مهندس لمهندس" قبل التنفيذ

### 1. CSS Logical Properties
دي **أهم نقطة** في الـ RTL. لو الوكيل كتب `margin-left`، قله **"غلط"**. لازم يستخدم `margin-inline-start`. دي الطريقة الحديثة اللي بتخلي الـ CSS يقلب لوحده لما اللغة تتغير من عربي لإنجليزي والعكس.

### 2. الخطوط (Fonts)
في ملف الـ SCSS، خليه يستخدم خطوط زي **"Tajawal"** أو **"Cairo"** من Google Fonts للعربي، و **"Roboto"** للإنجليزي. ده بيفرق جداً في شكل التطبيق الاحترافي ("رصين وجاد" زي ما طلبت).

### 3. خرائط RTL
خد بالك إن أزرار التحكم في الخريطة (الزووم وغيره) ممكن مكانها يبوظ في الـ RTL. اطلب منه يتأكد إن الـ CSS بتاع الخريطة **معزول (Isolated)** أو مظبوط عليه `dir="ltr"` لو الخريطة نفسها إنجليزي، أو متظبطة صح للعربي.

---

## المراحل والأولويات

### المرحلة 1: البنية التحتية
- [ ] إعداد مشروع Angular 18+ بوضع Standalone
- [ ] إعداد مشروع FastAPI مع هيكل DDD
- [ ] تكوين PostgreSQL + PostGIS
- [ ] إعداد نظام المصادقة JWT

### المرحلة 2: الميزات الأساسية
- [ ] نظام تسجيل الدخول عبر QR
- [ ] محرك التمويل (Finance Engine)
- [ ] خدمة الموقع الجغرافي (GeoService)
- [ ] نظام إدارة المستخدمين (VIP/Standard)

### المرحلة 3: واجهة المستخدم
- [ ] تنفيذ نظام RTL الكامل
- [ ] مبدل اللغة (Language Switcher)
- [ ] لوحة القيادة (Dashboard)
- [ ] مكون التتبع مع الخريطة

### المرحلة 4: التحسينات
- [ ] اختبارات الوحدة والتكامل
- [ ] اختبارات RTL/LTR
- [ ] تحسين الأداء
