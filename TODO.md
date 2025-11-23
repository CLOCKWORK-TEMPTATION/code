plan.md
Architecture Overview
┌─────────────────────────────────────────────────────────────┐
│ Angular 18+ Super-App │
├─────────────────────────────────────────────────────────────┤
│ app/ │
│ ├── core/ (Singletons: Auth, API, Gemini) │
│ ├── shared/ (UI Components, Pipes, Directives) │
│ ├── domain/ (Drama Engine - Pure TS Logic) │
│ └── features/ (12 Lazy-Loaded Studios) │
│ ├── directors-studio/ [PRIORITY 1] │
│ ├── cinematography/ │
│ ├── creative-writing/ │
│ ├── prompt-studio/ │
│ ├── actor-ai/ [Complete Rewrite] │
│ ├── editor/ [Complex DOM Logic] │
│ ├── analysis/ [Seven Stations Viz] │
│ ├── brainstorm/ │
│ ├── breakdown/ │
│ ├── development/ │
│ ├── metrics/ │
│ └── dashboard/ │
└─────────────────────────────────────────────────────────────┘
Phase 1: Foundation Setup
1.1 Project Scaffolding
Task Technology Configuration
Initialize Angular ng new v18+ Standalone components, SSR optional
Tailwind Setup v4 Port existing globals.css variables
TypeScript Config Strict mode Match existing tsconfig.json rules
Testing Framework Jest + Playwright Port existing test configs
Linting ESLint + Prettier Match current code style
1.2 Directory Structure Creation
src/app/
├── core/
│ ├── services/
│ │ ├── auth.service.ts
│ │ ├── api.service.ts
│ │ └── gemini.service.ts
│ ├── guards/
│ │ └── auth.guard.ts
│ └── interceptors/
│ └── http-error.interceptor.ts
├── shared/
│ ├── ui/ # Ported Shadcn components
│ ├── pipes/
│ └── directives/
├── domain/
│ └── drama-engine/ # The kernel - framework agnostic
│ ├── models/
│ ├── services/
│ └── types/
└── features/
└── [12 studio modules]
Phase 2: Core Engine Migration
2.1 Drama Analyst Engine Port
Source: src/lib/drama-analyst (Next.js)
Target: src/app/domain/drama-engine (Angular)

Component Migration Strategy Output
Analysis Logic Pure TS classes AnalysisEngine.ts
Seven Stations Injectable Service SevenStationsService.ts
Scene Parser Utility functions scene-parser.utils.ts
Character Tracker Signal-based Service CharacterStateService.ts
Key Principle: This layer must remain framework-agnostic. No Angular decorators in core domain logic—only in the service wrappers.

2.2 Signal Store Pattern
// Example: CharacterStateService
@Injectable({ providedIn: 'root' })
export class CharacterStateService {
// Private mutable state
#state = signal<CharacterState>({
characters: [],
activeCharacter: null
});

// Public readonly state
readonly state = this.#state.asReadonly();

// Computed values
readonly activeCharacterName = computed(() =>
this.state().activeCharacter?.name ?? 'None'
);

// Actions
updateCharacter(id: string, data: Partial<Character>) {
this.#state.update(state => ({
...state,
characters: state.characters.map(c =>
c.id === id ? { ...c, ...data } : c
)
}));
}
}
Phase 3: Shared UI Library
3.1 Component Porting Strategy
Shadcn Component Angular Equivalent Priority
Button <app-button> P0
Card <app-card> P0
Dialog <app-dialog> P0
Input <app-input> P0
Tabs <app-tabs> P1
Accordion <app-accordion> P1
Select <app-select> P1
Implementation: Use Angular CDK primitives (Overlay, Portal) for Dialog/Select. Port Tailwind classes directly.

3.2 Styling Architecture
/_ globals.css - Tailwind v4 with CSS variables _/
@import "tailwindcss";

@theme inline {
--color-primary: hsl(222.2 47.4% 11.2%);
--color-secondary: hsl(210 40% 96.1%);
/_ ... port all existing design tokens _/
}

/_ Custom utilities for Arabic RTL support _/
@utility rtl-support {
direction: rtl;
text-align: right;
}
Phase 4: Feature Migration (Vertical Slice)
4.1 Directors Studio (Proof of Concept)
Routes:

// features/directors-studio/directors-studio.routes.ts
export const routes: Routes = [
{
path: '',
component: DirectorsStudioLayoutComponent,
children: [
{ path: '', component: ProjectListComponent },
{ path: 'project/:id', component: ProjectDetailComponent },
{ path: 'scene/:id', component: SceneAnalysisComponent }
]
}
];
State Management:

DirectorsStudioService (Signal Store)
├── Projects State
├── Active Project State
└── Scene Analysis State
└── Injects: DramaEngineService
Components Breakdown:

directors-studio/
├── components/
│ ├── project-card/
│ ├── scene-timeline/
│ └── analysis-panel/
├── services/
│ └── directors-studio.service.ts
├── models/
│ └── project.model.ts
└── directors-studio.routes.ts
4.2 Actor AI (Complete Rewrite)
Current Issue: Static app.js with vanilla DOM manipulation
Solution: Angular component with reactive forms

@Component({
selector: 'app-actor-ai',
template: `
<div class="actor-ai-container">
<app-character-selector
[characters]="characters()"
(select)="onSelectCharacter($event)" />

      <app-dialogue-generator
        [character]="activeCharacter()"
        (generate)="onGenerate($event)" />
    </div>

`
})
export class ActorAiComponent {
characters = signal<Character[]>([]);
activeCharacter = signal<Character | null>(null);

constructor(private gemini: GeminiService) {}

async onGenerate(prompt: string) {
const response = await this.gemini.generateDialogue(
this.activeCharacter()!,
prompt
);
// Handle response with signals
}
}
4.3 Screenplay Editor (Complex DOM Logic)
Challenge: Rich text editing with Arabic formatting rules
Approach: Isolate into standalone library

@features/editor/
├── components/
│ ├── editor-toolbar/
│ ├── editor-canvas/ # Uses Renderer2 for DOM
│ └── format-panel/
├── services/
│ ├── editor-state.service.ts
│ └── text-formatter.service.ts
└── directives/
└── arabic-text-mask.directive.ts
Key Rule: NO document.getElementById. Use @ViewChild, Renderer2, or template refs.

Phase 5: Remaining Features (Pattern Replication)
5.1 Migration Order
Feature Complexity Dependencies Estimated Effort
Dashboard Low Core only 2 days
Metrics Low Core + Charts 3 days
Brainstorm Medium Drama Engine 4 days
Breakdown Medium Drama Engine 4 days
Development Medium Drama Engine 4 days
Analysis (7 Stations) High Drama Engine + D3 5 days
Cinematography High Drama Engine + Canvas 5 days
Creative Writing High Gemini Service 5 days
Prompt Studio Medium Gemini Service 4 days
5.2 Lazy Loading Configuration
// app.routes.ts
export const routes: Routes = [
{
path: 'directors-studio',
loadChildren: () => import('./features/directors-studio/directors-studio.routes')
.then(m => m.routes)
},
{
path: 'cinematography',
loadChildren: () => import('./features/cinematography/cinematography.routes')
.then(m => m.routes)
},
// ... repeat for all 12 features
{ path: '', redirectTo: '/dashboard', pathMatch: 'full' },
{ path: '**', component: NotFoundComponent }
];
Phase 6: Testing Strategy
6.1 Test Coverage Requirements
Layer Framework Coverage Target
Services (Core) Jest 90%+
Components (UI) Angular Testing Library 80%+
Integration Jest + TestBed 70%+
E2E (Critical Flows) Playwright 100% of user journeys
6.2 Test Structure
src/
├── app/
│ └── [features]/
│ ├── _.component.spec.ts
│ └── _.service.spec.ts
└── tests/
├── e2e/
│ ├── directors-studio.spec.ts
│ └── actor-ai.spec.ts
└── integration/
└── drama-engine.spec.ts
Phase 7: Performance & Optimization
7.1 Bundle Size Optimization
Lazy Loading: All 12 features loaded on-demand
Tree Shaking: Ensure pure TS in domain layer
Image Optimization: Port existing optimize-images.js script
Code Splitting: Separate vendor chunks for Tailwind, Angular Material
7.2 Performance Budget
Port existing performance-budget.json and lighthouserc.json to Angular build pipeline.

Phase 8: Deployment Pipeline
8.1 Build Configuration
// angular.json (production config)
{
"configurations": {
"production": {
"optimization": true,
"outputHashing": "all",
"sourceMap": false,
"namedChunks": false,
"aot": true,
"buildOptimizer": true,
"budgets": [
{
"type": "initial",
"maximumWarning": "500kb",
"maximumError": "1mb"
}
]
}
}
}
8.2 Cutover Strategy
Parallel Development: Build at v2.platform.com or localhost:4200
Feature Parity Checklist: Ensure all 12 studios functional
DNS Switch: Point production domain to Angular app
Rollback Plan: Keep Next.js build available for 48h
Definition of Done
Angular 18+ project scaffolded with Tailwind v4
Drama Engine ported to src/app/domain (framework-agnostic)
Core services (Auth, API, Gemini) implemented with Signal Stores
Shared UI library created with ported Shadcn components
Directors Studio fully migrated and functional (proof of concept)
All 12 features lazy-loaded with proper routing
Actor AI rewritten from vanilla JS to Angular components
Screenplay Editor isolated with Renderer2 (no direct DOM)
Test coverage: 90% services, 80% components, 100% E2E critical flows
Performance budget met (Lighthouse score ≥ 90)
Production build optimized (bundle size < 1MB initial)
Deployment pipeline configured with rollback plan
To-dos (8)
Scaffold Angular 18+: Initialize project, configure Tailwind v4, setup Jest + Playwright
Port Drama Engine: Migrate drama-analyst to src/app/domain as pure TypeScript
Build Core Services: Implement Auth, API, Gemini with Signal Store pattern
Create Shared UI: Port Shadcn components to Angular with CDK primitives
Migrate Directors Studio: Complete vertical slice as proof of concept
Replicate Pattern: Migrate remaining 11 features following established architecture
Implement Testing: Achieve 90% service, 80% component, 100% E2E coverage
Optimize & Deploy: Bundle optimization, performance budgets, production cutover
