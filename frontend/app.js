const API_URL = ['localhost:5173', '127.0.0.1:5173'].includes(window.location.host)
  ? 'http://127.0.0.1:8000'
  : '';
const fileInput = document.querySelector('#file-input');
const dropzone = document.querySelector('#dropzone');
const form = document.querySelector('#upload-form');
const analyzeButton = document.querySelector('#analyze-button');
const sourceImage = document.querySelector('#source-image');
const canvas = document.querySelector('#heatmap-canvas');
const errorMessage = document.querySelector('#error-message');
const heatmapToggle = document.querySelector('#heatmap-toggle');
const llmReportButton = document.querySelector('#llm-report-button');
const llmReport = document.querySelector('#llm-report');
const llmReportText = document.querySelector('#llm-report-text');
let selectedFile = null;
let latestAnalysis = null;

async function checkHealth() {
  try {
    const response = await fetch(`${API_URL}/api/health`);
    const health = await response.json();
    const statusText = health.status === 'demo_mode'
      ? 'Demo mode'
      : health.model_type === 'chexpert_multilabel'
        ? '14-finding model ready'
        : health.model_available ? 'Pneumonia model ready' : 'Model missing';
    document.querySelector('#status-text').textContent = statusText;
    if (health.model_available || health.status === 'demo_mode') document.querySelector('#status-dot').classList.add('ready');
  } catch (_) { document.querySelector('#status-text').textContent = 'API offline'; }
  try {
    const llmResponse = await fetch(`${API_URL}/api/llm-health`);
    const llmHealth = await llmResponse.json();
    llmReportButton.title = llmHealth.available && llmHealth.model_installed
      ? `Local report model: ${llmHealth.model}`
      : `Local LLM unavailable: ${llmHealth.model}`;
  } catch (_) {
    llmReportButton.title = 'Local LLM unavailable';
  }
}

function chooseFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  selectedFile = file;
  analyzeButton.disabled = false;
  document.querySelector('#filename').textContent = file.name;
  sourceImage.src = URL.createObjectURL(file);
  sourceImage.hidden = false;
  canvas.hidden = true;
  heatmapToggle.hidden = true;
  document.querySelector('.empty-stage').hidden = true;
  dropzone.classList.add('active');
  errorMessage.hidden = true;
}

fileInput.addEventListener('change', () => chooseFile(fileInput.files[0]));
['dragenter', 'dragover'].forEach(eventName => dropzone.addEventListener(eventName, event => { event.preventDefault(); dropzone.classList.add('active'); }));
['dragleave', 'drop'].forEach(eventName => dropzone.addEventListener(eventName, event => { event.preventDefault(); if (eventName === 'drop') chooseFile(event.dataTransfer.files[0]); else if (!selectedFile) dropzone.classList.remove('active'); }));

function renderHeatmap(values) {
  const context = canvas.getContext('2d');
  canvas.width = values[0].length;
  canvas.height = values.length;
  const flattenedValues = values.flat().map(value => Math.max(0, Math.min(1, value)));
  const sortedValues = [...flattenedValues].sort((left, right) => left - right);
  const low = sortedValues[Math.floor((sortedValues.length - 1) * 0.72)];
  const high = sortedValues[Math.floor((sortedValues.length - 1) * 0.99)];
  const range = Math.max(high - low, 1e-6);
  const imageData = context.createImageData(canvas.width, canvas.height);
  flattenedValues.forEach((value, index) => {
    const relativeValue = Math.max(0, Math.min(1, (value - low) / range));
    const intensity = Math.pow(relativeValue, 1.6);
    imageData.data[index * 4] = 255;
    imageData.data[index * 4 + 1] = Math.round(190 * (1 - intensity));
    imageData.data[index * 4 + 2] = 0;
    imageData.data[index * 4 + 3] = Math.round(intensity * 125);
  });
  context.putImageData(imageData, 0, 0);
  canvas.hidden = false;
  heatmapToggle.hidden = false;
  heatmapToggle.textContent = 'Hide heatmap';
  heatmapToggle.setAttribute('aria-pressed', 'true');
  fitHeatmapToImage();
}

function fitHeatmapToImage() {
  if (canvas.hidden || sourceImage.hidden) return;
  const imageRect = sourceImage.getBoundingClientRect();
  const stageRect = document.querySelector('#image-stage').getBoundingClientRect();
  if (!imageRect.width || !imageRect.height) return;
  Object.assign(canvas.style, {
    left: `${imageRect.left - stageRect.left}px`,
    top: `${imageRect.top - stageRect.top}px`,
    width: `${imageRect.width}px`,
    height: `${imageRect.height}px`,
  });
}

function renderResult(data) {
  latestAnalysis = data;
  llmReportButton.disabled = false;
  llmReport.hidden = true;
  llmReportText.textContent = '';
  const triage = data.triage;
  const priority = document.querySelector('#priority');
  priority.textContent = triage.priority;
  priority.className = triage.priority.toLowerCase();
  document.querySelector('#finding').textContent = triage.finding;
  document.querySelector('#risk').textContent = triage.risk.toFixed(3);
  document.querySelector('#confidence').textContent = `${(triage.confidence * 100).toFixed(1)}%`;
  document.querySelector('#severity').textContent = `${(triage.severity * 100).toFixed(0)}%`;
  document.querySelector('#result-time').textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  document.querySelector('#disclaimer').textContent = data.disclaimer;
  document.querySelector('#mode-label').textContent = data.mode === 'demo'
    ? 'DEMO OUTPUT - NOT A MEDICAL PREDICTION'
    : 'RESEARCH SUPPORT ONLY';
  const list = document.querySelector('#probabilities');
  list.innerHTML = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]).map(([name, value]) => `<div class="probability-row"><span>${name}</span><span>${(value * 100).toFixed(1)}%</span><div class="bar"><b style="width:${value * 100}%"></b></div></div>`).join('');
  renderHeatmap(data.heatmap);
}

heatmapToggle.addEventListener('click', () => {
  canvas.hidden = !canvas.hidden;
  heatmapToggle.textContent = canvas.hidden ? 'Show heatmap' : 'Hide heatmap';
  heatmapToggle.setAttribute('aria-pressed', String(!canvas.hidden));
});

sourceImage.addEventListener('load', fitHeatmapToImage);
window.addEventListener('resize', fitHeatmapToImage);

llmReportButton.addEventListener('click', async () => {
  if (!latestAnalysis) return;
  const triage = latestAnalysis.triage;
  llmReportButton.disabled = true;
  llmReportButton.textContent = 'Generating local review...';
  llmReport.hidden = false;
  llmReportText.textContent = 'Preparing a non-diagnostic report...';
  try {
    const response = await fetch(`${API_URL}/api/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        finding: triage.finding,
        confidence: triage.confidence,
        severity: triage.severity,
        risk: triage.risk,
        priority: triage.priority,
        mode: latestAnalysis.mode,
        probabilities: latestAnalysis.probabilities,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Review generation failed');
    llmReportText.textContent = `${result.report}\n\nGenerated locally with ${result.model}.`;
  } catch (error) {
    llmReportText.textContent = `Local review unavailable: ${error.message}`;
  } finally {
    llmReportButton.disabled = false;
    llmReportButton.textContent = 'Generate local review note';
  }
});

form.addEventListener('submit', async event => {
  event.preventDefault(); if (!selectedFile) return;
  analyzeButton.disabled = true; analyzeButton.textContent = 'Analyzing...'; errorMessage.hidden = true;
  const body = new FormData(); body.append('file', selectedFile);
  try {
    const response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Analysis failed');
    renderResult(data);
  } catch (error) { errorMessage.textContent = error.message; errorMessage.hidden = false; }
  finally { analyzeButton.disabled = false; analyzeButton.textContent = 'Analyze image'; }
});

checkHealth();
