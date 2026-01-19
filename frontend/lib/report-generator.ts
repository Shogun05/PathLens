import jsPDF from 'jspdf';

export interface AmenityPlacement {
    type: string;
    latitude: number;
    longitude: number;
    description?: string;
}

export interface MetricsData {
    network?: {
        circuity_sample_ratio?: number;
        intersection_density_global?: number;
        link_node_ratio_global?: number;
    };
    scores?: {
        citywide?: {
            accessibility_mean?: number;
            travel_time_min_mean?: number;
            walkability_mean?: number;
            node_count?: number;
        };
        underserved?: {
            accessibility_mean?: number;
            travel_time_min_mean?: number;
            walkability_mean?: number;
            node_count?: number;
            percentile_threshold?: number;
        };
        well_served?: {
            accessibility_mean?: number;
            travel_time_min_mean?: number;
            walkability_mean?: number;
            node_count?: number;
        };
        gap_closure?: {
            threshold_minutes?: number;
            nodes_above_threshold?: number;
            pct_above_threshold?: number;
            total_nodes?: number;
        };
        distribution?: {
            travel_time_p50?: number;
            travel_time_p90?: number;
            travel_time_p95?: number;
            travel_time_max?: number;
            accessibility_p10?: number;
            accessibility_p50?: number;
        };
        equity?: number;
    };
}

export interface ReportData {
    location: string;
    baselineScore: number;
    optimizedScore: number;
    amenities: AmenityPlacement[];
    generatedAt: Date;
    baselineMetrics?: MetricsData;
    optimizedMetrics?: MetricsData;
    optimizationMode?: string;
    optimizationResults?: {
        generation?: number;
        fitness?: number;
        placements?: Record<string, number>;
        amenityUtilities?: Record<string, number>;
    };
}

// Color palette matching PathLens branding
const COLORS = {
    primary: '#8fd6ff',
    dark: '#0f1c23',
    accent: '#10b981',
    text: '#333333',
    textLight: '#666666',
    border: '#e0e0e0',
    headerBg: '#1b2328',
    rowAlt: '#f8fafc',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
};

// Amenity type colors for visual distinction
const AMENITY_COLORS: Record<string, string> = {
    bank: '#3b82f6',
    hospital: '#ef4444',
    school: '#f59e0b',
    pharmacy: '#8b5cf6',
    supermarket: '#10b981',
    bus_station: '#06b6d4',
    park: '#22c55e',
};

/**
 * Generate a professional PDF report for PathLens optimization results
 */
export async function generateOptimizationReport(
    mapContainer: HTMLElement | null,
    data: ReportData
): Promise<void> {
    const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
    });

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pageWidth - margin * 2;

    let yPos = margin;

    // === PAGE 1: Executive Summary ===
    yPos = drawHeader(pdf, data, pageWidth, margin);

    // === Key Metrics Overview ===
    yPos = drawExecutiveSummary(pdf, data, margin, contentWidth, yPos);

    // === Score Comparison ===
    yPos = drawScoreComparison(pdf, data, margin, contentWidth, yPos);

    // Footer for page 1
    drawPageFooter(pdf, pageWidth, pageHeight);

    // === PAGE 2: Detailed Amenity Table ===
    pdf.addPage();
    yPos = margin;

    // Table header
    pdf.setFillColor(27, 35, 40);
    pdf.rect(0, 0, pageWidth, 25, 'F');
    pdf.setTextColor(143, 214, 255);
    pdf.setFontSize(16);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Proposed Amenity Placements', margin, 16);
    yPos = 35;

    // Draw amenity table
    yPos = drawAmenityTable(pdf, data, margin, contentWidth, yPos, pageWidth, pageHeight);

    // Footer on last page
    drawPageFooter(pdf, pageWidth, pageHeight);

    // Save the PDF
    const filename = `pathlens-report-${data.location?.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'optimization'}-${data.generatedAt.toISOString().slice(0, 10)}.pdf`;
    pdf.save(filename);
}

// Add footer to each page
function drawPageFooter(pdf: jsPDF, pageWidth: number, pageHeight: number): void {
    pdf.setFontSize(8);
    pdf.setTextColor(150, 150, 150);
    pdf.text('©PathLens', pageWidth / 2, pageHeight - 8, { align: 'center' });
}

function drawHeader(pdf: jsPDF, data: ReportData, pageWidth: number, margin: number): number {
    // Header background
    pdf.setFillColor(27, 35, 40);
    pdf.rect(0, 0, pageWidth, 50, 'F');

    // Logo/Title
    pdf.setTextColor(143, 214, 255);
    pdf.setFontSize(28);
    pdf.setFont('helvetica', 'bold');
    pdf.text('PATHLENS', margin, 22);

    pdf.setFontSize(12);
    pdf.setTextColor(255, 255, 255);
    pdf.text('Urban Accessibility Optimization Report', margin, 32);

    // Optimization mode badge
    if (data.optimizationMode) {
        pdf.setFontSize(9);
        pdf.setTextColor(16, 185, 129);
        pdf.text(`Mode: ${data.optimizationMode}`, margin, 42);
    }

    // Metadata on right side
    pdf.setFontSize(10);
    pdf.setTextColor(200, 200, 200);
    const dateStr = data.generatedAt.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
    });
    pdf.text(dateStr, pageWidth - margin, 22, { align: 'right' });
    pdf.text(data.location || 'Urban Analysis', pageWidth - margin, 32, { align: 'right' });

    return 60;
}

function drawExecutiveSummary(pdf: jsPDF, data: ReportData, margin: number, contentWidth: number, yPos: number): number {
    pdf.setFontSize(14);
    pdf.setTextColor(51, 51, 51);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Executive Summary', margin, yPos);
    yPos += 8;

    const improvement = data.optimizedScore - data.baselineScore;
    const improvementPct = data.baselineScore > 0 ? ((improvement / data.baselineScore) * 100) : 0;

    // Summary box
    pdf.setFillColor(248, 250, 252);
    pdf.setDrawColor(200, 200, 200);
    pdf.roundedRect(margin, yPos, contentWidth, 45, 3, 3, 'FD');

    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(80, 80, 80);

    const summaryText = `This optimization analysis for ${data.location || 'the selected area'} evaluated ${data.amenities.length} potential amenity placements across ${Object.keys(AMENITY_COLORS).length} categories. The proposed interventions are projected to improve the citywide accessibility score from ${data.baselineScore.toFixed(1)} to ${data.optimizedScore.toFixed(1)}, representing a ${improvementPct.toFixed(1)}% improvement.`;

    const lines = pdf.splitTextToSize(summaryText, contentWidth - 10);
    pdf.text(lines, margin + 5, yPos + 8);

    // Key stats row
    yPos += 25;
    const statsY = yPos + 10;

    // Stat 1: Total Placements
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(51, 51, 51);
    pdf.setFontSize(18);
    pdf.text(data.amenities.length.toString(), margin + 25, statsY, { align: 'center' });
    pdf.setFontSize(8);
    pdf.setTextColor(100, 100, 100);
    pdf.setFont('helvetica', 'normal');
    pdf.text('Total Placements', margin + 25, statsY + 6, { align: 'center' });

    // Stat 2: Improvement
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(16, 185, 129);
    pdf.setFontSize(18);
    pdf.text(`+${improvementPct.toFixed(1)}%`, margin + contentWidth / 2, statsY, { align: 'center' });
    pdf.setFontSize(8);
    pdf.setTextColor(100, 100, 100);
    pdf.setFont('helvetica', 'normal');
    pdf.text('Score Improvement', margin + contentWidth / 2, statsY + 6, { align: 'center' });

    // Stat 3: Categories
    const categories = new Set(data.amenities.map(a => a.type.toLowerCase())).size;
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(51, 51, 51);
    pdf.setFontSize(18);
    pdf.text(categories.toString(), margin + contentWidth - 25, statsY, { align: 'center' });
    pdf.setFontSize(8);
    pdf.setTextColor(100, 100, 100);
    pdf.setFont('helvetica', 'normal');
    pdf.text('Amenity Types', margin + contentWidth - 25, statsY + 6, { align: 'center' });

    return yPos + 35;
}

function drawScoreComparison(pdf: jsPDF, data: ReportData, margin: number, contentWidth: number, yPos: number): number {
    pdf.setFontSize(12);
    pdf.setTextColor(51, 51, 51);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Accessibility Score Comparison', margin, yPos);
    yPos += 8;

    const boxHeight = 35;
    const gap = 10; // Small gap between boxes
    const boxWidth = (contentWidth - gap) / 2;

    // Baseline box
    pdf.setFillColor(255, 245, 245);
    pdf.setDrawColor(239, 68, 68);
    pdf.setLineWidth(0.5);
    pdf.roundedRect(margin, yPos, boxWidth, boxHeight, 3, 3, 'FD');

    pdf.setFontSize(9);
    pdf.setTextColor(100, 100, 100);
    pdf.setFont('helvetica', 'normal');
    pdf.text('BASELINE', margin + boxWidth / 2, yPos + 10, { align: 'center' });

    pdf.setFontSize(20);
    pdf.setTextColor(239, 68, 68);
    pdf.setFont('helvetica', 'bold');
    pdf.text(`${data.baselineScore.toFixed(1)}/100`, margin + boxWidth / 2, yPos + 25, { align: 'center' });

    // Optimized box
    const optX = margin + boxWidth + gap;
    pdf.setFillColor(240, 253, 244);
    pdf.setDrawColor(16, 185, 129);
    pdf.roundedRect(optX, yPos, boxWidth, boxHeight, 3, 3, 'FD');

    pdf.setFontSize(9);
    pdf.setTextColor(100, 100, 100);
    pdf.setFont('helvetica', 'normal');
    pdf.text('OPTIMIZED', optX + boxWidth / 2, yPos + 10, { align: 'center' });

    pdf.setFontSize(20);
    pdf.setTextColor(16, 185, 129);
    pdf.setFont('helvetica', 'bold');
    pdf.text(`${data.optimizedScore.toFixed(1)}/100`, optX + boxWidth / 2, yPos + 25, { align: 'center' });

    return yPos + boxHeight + 10;
}

function drawAmenityDistribution(pdf: jsPDF, data: ReportData, margin: number, contentWidth: number, yPos: number, pageHeight: number): number {
    if (yPos > pageHeight - 80) {
        pdf.addPage();
        yPos = 20;
    }

    pdf.setFontSize(12);
    pdf.setTextColor(51, 51, 51);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Amenity Distribution', margin, yPos);
    yPos += 8;

    // Count amenities by type
    const amenitiesByType = data.amenities.reduce((acc, a) => {
        const type = a.type.toLowerCase();
        acc[type] = (acc[type] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

    const typeEntries = Object.entries(amenitiesByType).sort((a, b) => b[1] - a[1]);
    const maxCount = Math.max(...typeEntries.map(([, count]) => count));

    // Draw horizontal bar chart
    const barHeight = 8;
    const labelWidth = 35;
    const barMaxWidth = contentWidth - labelWidth - 25;

    typeEntries.forEach(([type, count], index) => {
        const rowY = yPos + index * (barHeight + 4);
        const barWidth = (count / maxCount) * barMaxWidth;
        const color = AMENITY_COLORS[type] || '#666666';

        // Label
        pdf.setFontSize(8);
        pdf.setFont('helvetica', 'normal');
        pdf.setTextColor(51, 51, 51);
        const displayName = type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        pdf.text(displayName, margin, rowY + 5.5);

        // Bar
        pdf.setFillColor(
            parseInt(color.slice(1, 3), 16),
            parseInt(color.slice(3, 5), 16),
            parseInt(color.slice(5, 7), 16)
        );
        pdf.roundedRect(margin + labelWidth, rowY, barWidth, barHeight, 1, 1, 'F');

        // Count
        pdf.setFontSize(8);
        pdf.setFont('helvetica', 'bold');
        pdf.text(count.toString(), margin + labelWidth + barWidth + 3, rowY + 5.5);
    });

    return yPos + typeEntries.length * (barHeight + 4) + 10;
}

function drawNetworkAnalysis(pdf: jsPDF, data: ReportData, margin: number, contentWidth: number, yPos: number): number {
    pdf.setFontSize(12);
    pdf.setTextColor(51, 51, 51);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Network Characteristics', margin, yPos);
    yPos += 8;

    const network = data.optimizedMetrics?.network || {};

    pdf.setFillColor(248, 250, 252);
    pdf.setDrawColor(200, 200, 200);
    pdf.roundedRect(margin, yPos, contentWidth, 30, 3, 3, 'FD');

    const col1 = margin + 10;
    const col2 = margin + contentWidth / 3 + 5;
    const col3 = margin + (2 * contentWidth) / 3;

    pdf.setFontSize(9);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);
    pdf.text('Circuity Ratio', col1, yPos + 10);
    pdf.text('Intersection Density', col2, yPos + 10);
    pdf.text('Link-Node Ratio', col3, yPos + 10);

    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(51, 51, 51);
    pdf.setFontSize(12);
    pdf.text((network.circuity_sample_ratio || 0).toFixed(2), col1, yPos + 22);
    pdf.text((network.intersection_density_global || 0).toFixed(1), col2, yPos + 22);
    pdf.text((network.link_node_ratio_global || 0).toFixed(2), col3, yPos + 22);

    return yPos + 40;
}

function drawTravelTimeAnalysis(pdf: jsPDF, data: ReportData, margin: number, contentWidth: number, yPos: number): number {
    pdf.setFontSize(12);
    pdf.setTextColor(51, 51, 51);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Travel Time Distribution', margin, yPos);
    yPos += 8;

    const dist = data.optimizedMetrics?.scores?.distribution || {};
    const citywide = data.optimizedMetrics?.scores?.citywide || {};

    pdf.setFillColor(248, 250, 252);
    pdf.setDrawColor(200, 200, 200);
    pdf.roundedRect(margin, yPos, contentWidth, 40, 3, 3, 'FD');

    // Row 1
    const cols = [margin + 10, margin + contentWidth / 4, margin + contentWidth / 2, margin + (3 * contentWidth) / 4];

    pdf.setFontSize(8);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);
    pdf.text('Median (P50)', cols[0], yPos + 10);
    pdf.text('90th Percentile', cols[1], yPos + 10);
    pdf.text('95th Percentile', cols[2], yPos + 10);
    pdf.text('Maximum', cols[3], yPos + 10);

    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(51, 51, 51);
    pdf.setFontSize(11);
    pdf.text(`${(dist.travel_time_p50 || 0).toFixed(1)} min`, cols[0], yPos + 20);
    pdf.text(`${(dist.travel_time_p90 || 0).toFixed(1)} min`, cols[1], yPos + 20);
    pdf.text(`${(dist.travel_time_p95 || 0).toFixed(1)} min`, cols[2], yPos + 20);
    pdf.text(`${(dist.travel_time_max || 0).toFixed(1)} min`, cols[3], yPos + 20);

    // Row 2
    pdf.setFontSize(8);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);
    pdf.text('Mean Travel Time', cols[0], yPos + 32);
    pdf.text('Mean Walkability', cols[2], yPos + 32);

    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(51, 51, 51);
    pdf.setFontSize(10);
    pdf.text(`${(citywide.travel_time_min_mean || 0).toFixed(1)} min`, cols[0] + 40, yPos + 32);
    pdf.text(`${(citywide.walkability_mean || 0).toFixed(1)}/100`, cols[2] + 40, yPos + 32);

    return yPos + 50;
}

function drawEquityAnalysis(pdf: jsPDF, data: ReportData, margin: number, contentWidth: number, yPos: number): number {
    pdf.setFontSize(12);
    pdf.setTextColor(51, 51, 51);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Equity Analysis', margin, yPos);
    yPos += 8;

    const scores = data.optimizedMetrics?.scores || {};
    const underserved = scores.underserved || {};
    const wellServed = scores.well_served || {};
    const equity = scores.equity || 0;

    pdf.setFillColor(248, 250, 252);
    pdf.setDrawColor(200, 200, 200);
    pdf.roundedRect(margin, yPos, contentWidth, 50, 3, 3, 'FD');

    // Equity score
    pdf.setFontSize(9);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);
    pdf.text('Equity Score', margin + 25, yPos + 10, { align: 'center' });

    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(20);
    pdf.setTextColor(equity > 80 ? 16 : equity > 60 ? 245 : 239, equity > 80 ? 185 : equity > 60 ? 158 : 68, equity > 80 ? 129 : equity > 60 ? 11 : 68);
    pdf.text(`${equity.toFixed(0)}%`, margin + 25, yPos + 28, { align: 'center' });

    // Underserved stats
    const colU = margin + contentWidth / 3;
    pdf.setFontSize(8);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(239, 68, 68);
    pdf.text('UNDERSERVED AREAS', colU, yPos + 10);

    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(80, 80, 80);
    pdf.text(`Nodes: ${(underserved.node_count || 0).toLocaleString()}`, colU, yPos + 20);
    pdf.text(`Accessibility: ${(underserved.accessibility_mean || 0).toFixed(1)}`, colU, yPos + 28);
    pdf.text(`Travel Time: ${(underserved.travel_time_min_mean || 0).toFixed(1)} min`, colU, yPos + 36);

    // Well-served stats
    const colW = margin + (2 * contentWidth) / 3;
    pdf.setFontSize(8);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(16, 185, 129);
    pdf.text('WELL-SERVED AREAS', colW, yPos + 10);

    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(80, 80, 80);
    pdf.text(`Nodes: ${(wellServed.node_count || 0).toLocaleString()}`, colW, yPos + 20);
    pdf.text(`Accessibility: ${(wellServed.accessibility_mean || 0).toFixed(1)}`, colW, yPos + 28);
    pdf.text(`Travel Time: ${(wellServed.travel_time_min_mean || 0).toFixed(1)} min`, colW, yPos + 36);

    return yPos + 60;
}

function drawGapClosureAnalysis(pdf: jsPDF, data: ReportData, margin: number, contentWidth: number, yPos: number, pageHeight: number): number {
    if (yPos > pageHeight - 60) {
        pdf.addPage();
        yPos = 20;
    }

    const gapClosure = data.optimizedMetrics?.scores?.gap_closure || {};

    pdf.setFontSize(12);
    pdf.setTextColor(51, 51, 51);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Gap Closure Analysis', margin, yPos);
    yPos += 8;

    pdf.setFillColor(248, 250, 252);
    pdf.setDrawColor(200, 200, 200);
    pdf.roundedRect(margin, yPos, contentWidth, 35, 3, 3, 'FD');

    const threshold = gapClosure.threshold_minutes || 15;
    const nodesAbove = gapClosure.nodes_above_threshold || 0;
    const totalNodes = gapClosure.total_nodes || 0;
    const pctAbove = gapClosure.pct_above_threshold || 0;

    pdf.setFontSize(9);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);
    pdf.text(`Travel Time Threshold: ${threshold} minutes`, margin + 10, yPos + 12);

    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(239, 68, 68);
    pdf.setFontSize(16);
    pdf.text(`${pctAbove.toFixed(1)}%`, margin + contentWidth / 2, yPos + 20, { align: 'center' });

    pdf.setFontSize(9);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(80, 80, 80);
    pdf.text(`of nodes (${nodesAbove.toLocaleString()} / ${totalNodes.toLocaleString()}) exceed threshold`, margin + contentWidth / 2, yPos + 28, { align: 'center' });

    return yPos + 45;
}

function drawAmenityTable(pdf: jsPDF, data: ReportData, margin: number, contentWidth: number, yPos: number, pageWidth: number, pageHeight: number): number {
    const colWidths = { type: 50, lat: 55, lon: 55 };

    // Table header
    pdf.setFillColor(240, 240, 240);
    pdf.rect(margin, yPos, contentWidth, 8, 'F');

    pdf.setFontSize(9);
    pdf.setTextColor(80, 80, 80);
    pdf.setFont('helvetica', 'bold');

    let xPos = margin + 3;
    pdf.text('Type', xPos, yPos + 5.5);
    xPos += colWidths.type;
    pdf.text('Latitude', xPos, yPos + 5.5);
    xPos += colWidths.lat;
    pdf.text('Longitude', xPos, yPos + 5.5);

    yPos += 10;
    pdf.setFont('helvetica', 'normal');

    // Sort amenities by type
    const sortedAmenities = [...data.amenities].sort((a, b) =>
        a.type.localeCompare(b.type) || a.latitude - b.latitude
    );

    sortedAmenities.forEach((amenity, index) => {
        if (yPos > pageHeight - 20) {
            // Add footer to current page before creating new page
            pdf.setFontSize(8);
            pdf.setTextColor(150, 150, 150);
            pdf.text('©PathLens', pageWidth / 2, pageHeight - 8, { align: 'center' });
            
            pdf.addPage();
            yPos = margin;

            // Repeat header
            pdf.setFillColor(240, 240, 240);
            pdf.rect(margin, yPos, contentWidth, 8, 'F');
            pdf.setFontSize(9);
            pdf.setTextColor(80, 80, 80);
            pdf.setFont('helvetica', 'bold');

            let hxPos = margin + 3;
            pdf.text('Type', hxPos, yPos + 5.5);
            hxPos += colWidths.type;
            pdf.text('Latitude', hxPos, yPos + 5.5);
            hxPos += colWidths.lat;
            pdf.text('Longitude', hxPos, yPos + 5.5);

            yPos += 10;
            pdf.setFont('helvetica', 'normal');
        }

        // Alternating rows
        if (index % 2 === 1) {
            pdf.setFillColor(248, 250, 252);
            pdf.rect(margin, yPos - 3, contentWidth, 7, 'F');
        }

        pdf.setFontSize(8);
        pdf.setTextColor(51, 51, 51);

        let rowX = margin + 3;

        // Type with color
        const typeColor = AMENITY_COLORS[amenity.type.toLowerCase()] || '#666666';
        pdf.setFillColor(
            parseInt(typeColor.slice(1, 3), 16),
            parseInt(typeColor.slice(3, 5), 16),
            parseInt(typeColor.slice(5, 7), 16)
        );
        pdf.circle(rowX + 1, yPos + 0.5, 1.5, 'F');

        const displayType = amenity.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        pdf.text(displayType, rowX + 5, yPos + 2);
        rowX += colWidths.type;

        pdf.text(amenity.latitude.toFixed(6), rowX, yPos + 2);
        rowX += colWidths.lat;

        pdf.text(amenity.longitude.toFixed(6), rowX, yPos + 2);

        yPos += 7;
    });

    return yPos;
}

function drawFooter(pdf: jsPDF, data: ReportData, pageWidth: number, pageHeight: number): void {
    const dateStr = data.generatedAt.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
    });

    pdf.setFontSize(8);
    pdf.setTextColor(150, 150, 150);
    pdf.text(
        `Generated by PathLens • ${dateStr} • ${data.location}`,
        pageWidth / 2,
        pageHeight - 10,
        { align: 'center' }
    );
}

/**
 * Transform suggestions from the API into AmenityPlacement format
 */
export function transformSuggestionsToAmenities(
    suggestions: Array<{
        properties?: {
            id?: string;
            amenity_type?: string;
            amenity?: string;
        };
        geometry?: {
            coordinates?: [number, number];
        };
    }>
): AmenityPlacement[] {
    return suggestions
        .filter(s => s.geometry?.coordinates && s.properties)
        .map(s => ({
            type: s.properties?.amenity_type || s.properties?.amenity || 'unknown',
            longitude: s.geometry!.coordinates![0],
            latitude: s.geometry!.coordinates![1],
        }));
}
