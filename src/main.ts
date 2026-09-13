import { App } from './ui/app';

const app = new App(document.getElementById('app')!);
app.init();
(window as any).xorcery = app;
