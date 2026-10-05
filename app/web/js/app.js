/**
 * PaddleOCR Studio Frontend Interactive Application
 * Production Ingestion Pipeline with PP-OCRv6 Tiny CPU Support
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const processBtn = document.getElementById('process-btn');
    const langSelect = document.getElementById('lang-select');
    const deskewToggle = document.getElementById('deskew-toggle');
    const angleClsToggle = document.getElementById('angle-cls-toggle');

    const selectedFileInfo = document.getElementById('selected-file-info');
    const fileNameDisplay = document.getElementById('file-name-display');
    const fileSizeDisplay = document.getElementById('file-size-display');

    const viewportWrapper = document.getElementById('viewport-wrapper');
    const canvasContainer = document.getElementById('canvas-container');
    const canvasPlaceholder = document.getElementById('canvas-placeholder');
    const docSummaryBox = document.getElementById('doc-summary-box');
    const docTitle = document.getElementById('doc-title');
    const docMetaSub = document.getElementById('doc-meta-sub');
    const docPagesList = document.getElementById('doc-pages-list');
    const docIcon = document.getElementById('doc-icon');

    const sourceImage = document.getElementById('source-image');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const canvasTools = document.getElementById('canvas-tools');
    const boxTooltip = document.getElementById('box-tooltip');
    const showBoxesChk = document.getElementById('show-boxes-chk');

    // Metrics
    const metricTime = document.getElementById('metric-time');
    const metricLines = document.getElementById('metric-lines');
    const metricConf = document.getElementById('metric-conf');

    // Tab panes
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const outputTextArea = document.getElementById('output-text-area');
    const entitiesContainer = document.getElementById('entities-container');
    const tableRenderArea = document.getElementById('table-render-area');
    const jsonViewerPre = document.getElementById('json-viewer-pre');

    // Action buttons
    const copyTextBtn = document.getElementById('copy-text-btn');
    const downloadTxtBtn = document.getElementById('download-txt-btn');
    const copyJsonBtn = document.getElementById('copy-json-btn');
    const downloadJsonBtn = document.getElementById('download-json-btn');
    const downloadCsvBtn = document.getElementById('download-csv-btn');
    const downloadMdBtn = document.getElementById('download-md-btn');

    let currentFile = null;
    let documentResults = null;
    let hoveredBoxIndex = -1;

    // --- File Drag and Drop Handlers ---
    browseBtn.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('click', (e) => {
        if (e.target !== browseBtn) fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileSelection(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelection(e.target.files[0]);
    });

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function handleFileSelection(file) {
        currentFile = file;
        processBtn.disabled = false;
        
        // Show file banner
        selectedFileInfo.style.display = 'flex';
        fileNameDisplay.innerHTML = `<i class="fa-solid fa-file-lines" style="color: #06b6d4; margin-right: 8px;"></i> ${escapeHtml(file.name)}`;
        fileSizeDisplay.textContent = formatFileSize(file.size);

        const isImage = file.type.startsWith('image/');
        const isPdf = file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf';
        const isDocx = file.name.toLowerCase().endsWith('.docx') || file.name.toLowerCase().endsWith('.doc');

        if (isImage) {
            const reader = new FileReader();
            reader.onload = (e) => {
                sourceImage.src = e.target.result;
                sourceImage.onload = () => {
                    canvasPlaceholder.style.display = 'none';
                    docSummaryBox.style.display = 'none';
                    canvasContainer.style.display = 'inline-block';
                    canvasTools.style.display = 'flex';
                    setupCanvas();
                    showToast(`Loaded image "${file.name}" ready for OCR`);
                };
            };
            reader.readAsDataURL(file);
        } else {
            canvasContainer.style.display = 'none';
            canvasTools.style.display = 'none';
            canvasPlaceholder.style.display = 'none';
            docSummaryBox.style.display = 'block';

            if (isPdf) {
                docIcon.className = 'fa-solid fa-file-pdf';
                docIcon.style.color = '#ef4444';
            } else if (isDocx) {
                docIcon.className = 'fa-solid fa-file-word';
                docIcon.style.color = '#3b82f6';
            } else {
                docIcon.className = 'fa-solid fa-file-lines';
                docIcon.style.color = '#06b6d4';
            }

            docTitle.textContent = file.name;
            docMetaSub.textContent = `Size: ${formatFileSize(file.size)} • Click 'Run Ingestion' to extract`;
            docPagesList.innerHTML = '<p style="color: #94a3b8; font-size: 0.9rem;">Document ready for extraction.</p>';
            showToast(`Loaded document "${file.name}"`);
        }
    }

    // --- Tab Switching ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const targetPane = document.getElementById(target);
            if (targetPane) targetPane.classList.add('active');
        });
    });

    // --- Canvas & Box Drawing ---
    function setupCanvas() {
        const naturalW = sourceImage.naturalWidth || 800;
        const naturalH = sourceImage.naturalHeight || 600;
        
        overlayCanvas.width = naturalW;
        overlayCanvas.height = naturalH;
        
        drawBoxes();
    }

    function drawBoxes() {
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        const boxesToDraw = getActiveBoxes();
        if (!boxesToDraw || !showBoxesChk.checked) return;

        boxesToDraw.forEach((item, idx) => {
            const isHovered = idx === hoveredBoxIndex;
            const pts = item.box;
            if (!pts || pts.length < 4) return;

            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) {
                ctx.lineTo(pts[i][0], pts[i][1]);
            }
            ctx.closePath();

            // Confidence based coloring
            let strokeColor = '#06b6d4'; // Cyan
            let fillColor = 'rgba(6, 182, 212, 0.15)';
            if (item.confidence >= 0.90) {
                strokeColor = '#10b981'; // Emerald
                fillColor = isHovered ? 'rgba(16, 185, 129, 0.35)' : 'rgba(16, 185, 129, 0.15)';
            } else if (item.confidence < 0.75) {
                strokeColor = '#f43f5e'; // Rose
                fillColor = isHovered ? 'rgba(244, 63, 94, 0.35)' : 'rgba(244, 63, 94, 0.15)';
            }

            if (isHovered) {
                strokeColor = '#ffffff';
                ctx.lineWidth = 3;
            } else {
                ctx.lineWidth = 2;
            }

            ctx.strokeStyle = strokeColor;
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.stroke();
        });
    }

    function getActiveBoxes() {
        if (!documentResults) return null;
        if (documentResults.results && documentResults.results.length > 0) {
            return documentResults.results;
        }
        if (documentResults.pages && documentResults.pages.length > 0) {
            const firstPage = documentResults.pages[0];
            if (firstPage.results && firstPage.results.length > 0) {
                return firstPage.results;
            }
        }
        return null;
    }

    // --- Canvas Mouse Hover / Tooltip ---
    overlayCanvas.addEventListener('mousemove', (e) => {
        const boxes = getActiveBoxes();
        if (!boxes) return;

        const rect = overlayCanvas.getBoundingClientRect();
        const scaleX = overlayCanvas.width / rect.width;
        const scaleY = overlayCanvas.height / rect.height;

        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        let foundIdx = -1;
        for (let i = 0; i < boxes.length; i++) {
            const item = boxes[i];
            const pts = item.box;
            if (pts && isPointInPolygon([mouseX, mouseY], pts)) {
                foundIdx = i;
                break;
            }
        }

        if (foundIdx !== -1) {
            hoveredBoxIndex = foundIdx;
            const item = boxes[foundIdx];
            
            // Show Tooltip
            boxTooltip.style.display = 'block';
            boxTooltip.style.left = `${e.clientX - viewportWrapper.getBoundingClientRect().left + 15}px`;
            boxTooltip.style.top = `${e.clientY - viewportWrapper.getBoundingClientRect().top + 15}px`;
            document.getElementById('tt-conf').textContent = `${(item.confidence * 100).toFixed(1)}% Conf`;
            document.getElementById('tt-angle').textContent = `${item.angle || 0}° Angle`;
            document.getElementById('tt-text').textContent = item.text;
        } else {
            hoveredBoxIndex = -1;
            boxTooltip.style.display = 'none';
        }

        drawBoxes();
    });

    overlayCanvas.addEventListener('mouseleave', () => {
        hoveredBoxIndex = -1;
        boxTooltip.style.display = 'none';
        drawBoxes();
    });

    showBoxesChk.addEventListener('change', drawBoxes);

    function isPointInPolygon(point, vs) {
        const x = point[0], y = point[1];
        let inside = false;
        for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
            const xi = vs[i][0], yi = vs[i][1];
            const xj = vs[j][0], yj = vs[j][1];
            const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    // --- Process OCR Request ---
    processBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Document...';

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('lang', langSelect.value);
        formData.append('preprocess', deskewToggle.checked ? 'true' : 'false');
        formData.append('use_angle_cls', angleClsToggle.checked ? 'true' : 'false');

        try {
            const res = await fetch('/api/v1/ocr/upload', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error (HTTP ${res.status})`);
            }
            
            documentResults = await res.json();
            renderDocumentResults(documentResults);
            showToast('Document processed successfully!');

        } catch (err) {
            console.error(err);
            showToast(`Error: ${err.message}`, true);
        } finally {
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Ingestion & Extraction';
        }
    });

    function renderDocumentResults(data) {
        // Update metrics
        metricTime.textContent = `${data.processing_time_ms} ms`;
        
        let totalCount = 0;
        let avgConf = 0;
        let confSum = 0;
        let confCount = 0;

        if (data.pages && data.pages.length > 0) {
            totalCount = `${data.total_pages || data.pages.length} Pages`;
            data.pages.forEach(p => {
                if (p.confidence > 0) {
                    confSum += p.confidence;
                    confCount++;
                }
            });
            avgConf = confCount > 0 ? (confSum / confCount) : 1.0;
        } else if (data.results && data.results.length > 0) {
            totalCount = `${data.results.length} Lines`;
            avgConf = data.results.reduce((acc, it) => acc + it.confidence, 0) / data.results.length;
        } else {
            totalCount = `1 Doc`;
            avgConf = 1.0;
        }

        metricLines.textContent = totalCount;
        metricConf.textContent = `${(avgConf * 100).toFixed(1)}%`;

        // Text Pane
        outputTextArea.value = data.full_text || '';

        // Entities Pane
        renderEntities(data.entities || {});

        // Tables Pane
        renderTableData(data.tables || []);

        // JSON Pane
        jsonViewerPre.querySelector('code').textContent = JSON.stringify(data, null, 2);

        // Visualizer / Document Breakdown
        if (data.file_type === 'image' && canvasContainer.style.display !== 'none') {
            setupCanvas();
        } else if (data.pages && data.pages.length > 0) {
            docPagesList.innerHTML = '';
            data.pages.forEach(page => {
                const pdiv = document.createElement('div');
                pdiv.style.padding = '12px 16px';
                pdiv.style.background = 'rgba(255,255,255,0.03)';
                pdiv.style.border = '1px solid rgba(255,255,255,0.08)';
                pdiv.style.borderRadius = '8px';
                pdiv.style.marginBottom = '8px';

                const badgeColor = page.extraction_method.includes('ocr') ? '#06b6d4' : '#10b981';
                const methodLabel = page.extraction_method.includes('ocr') ? 'PP-OCRv6 Tiny (OCR)' : 'Digital Text (Native)';

                pdiv.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <strong style="color: #f8fafc; font-size: 0.9rem;">Page ${page.page_number}</strong>
                        <span style="font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; background: rgba(6,182,212,0.15); color: ${badgeColor}; border: 1px solid ${badgeColor}40;">${methodLabel}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; max-height: 80px; overflow: hidden; text-overflow: ellipsis; white-space: pre-wrap;">${escapeHtml(page.text.substring(0, 200))}${page.text.length > 200 ? '...' : ''}</div>
                `;
                docPagesList.appendChild(pdiv);
            });
        }
    }

    function renderEntities(entities) {
        entitiesContainer.innerHTML = '';
        const keys = Object.keys(entities);
        let count = 0;

        keys.forEach(k => {
            const val = entities[k];
            if (!val || (Array.isArray(val) && val.length === 0)) return;

            count++;
            const row = document.createElement('div');
            row.className = 'entity-row';
            const displayVal = Array.isArray(val) ? val.join(', ') : val;
            row.innerHTML = `
                <span class="entity-key">${escapeHtml(k.replace(/_/g, ' '))}</span>
                <span class="entity-val">${escapeHtml(String(displayVal))}</span>
            `;
            entitiesContainer.appendChild(row);
        });

        if (count === 0) {
            entitiesContainer.innerHTML = '<p class="empty-state-text">No structured business entities identified in document.</p>';
        }
    }

    function renderTableData(tables) {
        if (!tables || tables.length === 0 || tables[0].rows === 0) {
            tableRenderArea.innerHTML = '<p class="empty-state-text">No tabular structures detected in document.</p>';
            return;
        }

        let fullHtml = '';
        tables.forEach((table, tIdx) => {
            let html = `<div style="margin-bottom: 20px;"><div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;"><i class="fa-solid fa-table"></i> Table ${tIdx + 1} (${table.rows} rows × ${table.cols} cols)</div><table class="custom-table">`;
            
            if (table.headers && table.headers.length > 0) {
                html += '<thead><tr>';
                table.headers.forEach(h => html += `<th>${escapeHtml(h)}</th>`);
                html += '</tr></thead>';
            }

            html += '<tbody>';
            const bodyRows = table.headers && table.headers.length > 0 ? table.matrix.slice(1) : table.matrix;
            bodyRows.forEach(row => {
                html += '<tr>';
                row.forEach(cell => html += `<td>${escapeHtml(cell)}</td>`);
                html += '</tr>';
            });
            html += '</tbody></table></div>';
            fullHtml += html;
        });

        tableRenderArea.innerHTML = fullHtml;
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // --- Clipboard & Downloads ---
    copyTextBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(outputTextArea.value);
        showToast('Text copied to clipboard!');
    });

    copyJsonBtn.addEventListener('click', () => {
        if (documentResults) {
            navigator.clipboard.writeText(JSON.stringify(documentResults, null, 2));
            showToast('JSON copied to clipboard!');
        }
    });

    downloadTxtBtn.addEventListener('click', () => {
        downloadFile('extracted_text.txt', outputTextArea.value, 'text/plain');
    });

    downloadJsonBtn.addEventListener('click', () => {
        if (documentResults) {
            downloadFile('ocr_document_results.json', JSON.stringify(documentResults, null, 2), 'application/json');
        }
    });

    downloadCsvBtn.addEventListener('click', () => {
        if (documentResults && documentResults.tables && documentResults.tables.length > 0) {
            const combinedCsv = documentResults.tables.map(t => t.csv).join('\n\n');
            downloadFile('extracted_tables.csv', combinedCsv, 'text/csv');
        } else {
            showToast('No tabular data to export.', true);
        }
    });

    downloadMdBtn.addEventListener('click', () => {
        if (documentResults && documentResults.tables && documentResults.tables.length > 0) {
            const combinedMd = documentResults.tables.map(t => t.markdown).join('\n\n');
            downloadFile('extracted_tables.md', combinedMd, 'text/markdown');
        } else {
            showToast('No tabular data to export.', true);
        }
    });

    function downloadFile(filename, content, mime) {
        const blob = new Blob([content], { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    // --- Toast Notifications ---
    function showToast(msg, isError = false) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        if (isError) toast.style.borderColor = '#f43f5e';
        toast.textContent = msg;
        document.getElementById('toast-container').appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }
});
