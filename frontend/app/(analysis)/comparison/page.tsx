'use client';

import { useState, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { pathLensAPI } from '@/lib/api';
import MapComponent from '@/components/MapComponent';
import { ArrowLeft, TrendingUp, TrendingDown, Minus, Layers, BarChart3, MapPin, ChevronDown } from 'lucide-react';

interface ModeData {
    name: string;
    available: boolean;
    metrics: {
        scores?: {
            accessibility_mean?: number;
            walkability_mean?: number;
            travel_time_min_mean?: number;
        };
    };
    improvements: {
        [key: string]: {
            absolute: number;
            percent: number;
        };
    };
}

interface ComparisonData {
    city: string;
    baseline: {
        scores?: {
            accessibility_mean?: number;
            walkability_mean?: number;
            travel_time_min_mean?: number;
        };
    };
    modes: {
        [modeId: string]: ModeData;
    };
}

interface POIFeature {
    type: string;
    geometry: {
        type: string;
        coordinates: [number, number];
    };
    properties: {
        amenity?: string;
        osmid?: string;
        optimization_mode?: string;
        mode_name?: string;
    };
}

const MODE_COLORS: { [key: string]: string } = {
    ga_only: '#10b981',      // Emerald
    ga_milp: '#3b82f6',      // Blue
    ga_milp_pnmlr: '#8b5cf6', // Violet
};

const MODE_ORDER = ['ga_only', 'ga_milp', 'ga_milp_pnmlr'];

const CITIES = [
    { id: 'bangalore', name: 'Bengaluru', center: [12.9716, 77.5946] as [number, number] },
    { id: 'mumbai', name: 'Mumbai', center: [19.076, 72.8777] as [number, number] },
    { id: 'navi_mumbai', name: 'Navi Mumbai', center: [19.033, 73.0297] as [number, number] },
    { id: 'chandigarh', name: 'Chandigarh', center: [30.7333, 76.7794] as [number, number] },
];

export default function ComparisonPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [comparison, setComparison] = useState<ComparisonData | null>(null);
    const [selectedModes, setSelectedModes] = useState<Set<string>>(new Set());
    const [allPois, setAllPois] = useState<{ [modeId: string]: POIFeature[] }>({});
    // Read initial city from sessionStorage, default to bangalore
    const [selectedCity, setSelectedCity] = useState(() => {
        if (typeof window !== 'undefined') {
            return sessionStorage.getItem('selectedCity') || 'bangalore';
        }
        return 'bangalore';
    });
    const [cityDropdownOpen, setCityDropdownOpen] = useState(false);

    const currentCity = CITIES.find(c => c.id === selectedCity) || CITIES[0];

    useEffect(() => {
        const loadData = async () => {
            try {
                setLoading(true);
                setError(null);
                setSelectedModes(new Set()); // Reset selected modes on city change
                setAllPois({}); // Clear POIs

                // Fetch comparison data for selected city
                const comparisonData = await pathLensAPI.getModesComparison(selectedCity);
                setComparison(comparisonData);

                // Fetch POIs for each available mode
                const poisByMode: { [modeId: string]: POIFeature[] } = {};
                for (const modeId of MODE_ORDER) {
                    if (comparisonData.modes[modeId]?.available) {
                        try {
                            const poisData = await pathLensAPI.getModePois(modeId, selectedCity);
                            poisByMode[modeId] = poisData.features || [];
                        } catch (e) {
                            console.error(`Failed to load POIs for ${modeId}:`, e);
                            poisByMode[modeId] = [];
                        }
                    }
                }
                setAllPois(poisByMode);

            } catch (err) {
                console.error('Failed to load comparison data:', err);
                setError('Failed to load comparison data');
            } finally {
                setLoading(false);
            }
        };

        loadData();
    }, [selectedCity]);

    const toggleMode = (modeId: string) => {
        setSelectedModes(prev => {
            const next = new Set(prev);
            if (next.has(modeId)) {
                next.delete(modeId);
            } else {
                next.add(modeId);
            }
            return next;
        });
    };

    // Combine POIs from selected modes for map display - memoized for performance
    const combinedSuggestions = useMemo(() => {
        return Object.entries(allPois)
            .filter(([modeId]) => selectedModes.has(modeId))
            .flatMap(([modeId, pois]) =>
                pois.map(poi => ({
                    geometry: poi.geometry,
                    properties: {
                        id: `${modeId}-${poi.properties.osmid}`,
                        amenity_type: poi.properties.amenity || 'unknown',
                        description: `${poi.properties.mode_name}: ${poi.properties.amenity}`,
                        mode: modeId,
                    }
                }))
            );
    }, [allPois, selectedModes]);

    // Create a key for MapComponent based on selected modes to force remount
    const mapKey = useMemo(() => {
        return `map-${selectedCity}-${Array.from(selectedModes).sort().join('-')}`;
    }, [selectedCity, selectedModes]);

    const renderImprovementBadge = (improvement: { absolute: number; percent: number } | undefined, isLowerBetter: boolean = false) => {
        if (!improvement) return <Badge variant="outline" className="text-gray-400">N/A</Badge>;

        const isPositive = isLowerBetter ? improvement.percent > 0 : improvement.percent > 0;
        const Icon = improvement.percent > 0 ? TrendingUp : improvement.percent < 0 ? TrendingDown : Minus;

        return (
            <Badge
                variant="outline"
                className={`flex items-center gap-1 ${isPositive ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' :
                    improvement.percent < 0 ? 'text-red-400 border-red-500/30 bg-red-500/10' :
                        'text-gray-400 border-gray-500/30'
                    }`}
            >
                <Icon className="h-3 w-3" />
                {improvement.percent > 0 ? '+' : ''}{improvement.percent.toFixed(1)}%
            </Badge>
        );
    };

    if (loading) {
        return (
            <div className="flex h-full items-center justify-center bg-[#0a1014]">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[#8fd6ff] mx-auto mb-4"></div>
                    <p className="text-gray-400">Loading {currentCity.name} data...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-full flex-col pointer-events-none">
            {/* Header */}
            <div className="border-b border-white/10 bg-[#101518] px-6 py-3 z-20 pointer-events-auto">
                <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-4">
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-gray-400 hover:text-white"
                            onClick={() => router.push('/baseline')}
                        >
                            <ArrowLeft className="h-4 w-4" />
                        </Button>
                        <div className="flex flex-col gap-1">
                            <h1 className="text-lg font-bold text-white flex items-center gap-2">
                                <Layers className="h-5 w-5 text-[#8fd6ff]" />
                                Mode Comparison
                            </h1>
                            <p className="text-xs text-gray-400">Compare optimization strategies side-by-side</p>
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        {/* City Selector */}
                        <div className="relative">
                            <button
                                onClick={() => setCityDropdownOpen(!cityDropdownOpen)}
                                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#1b2328] border border-white/10 text-white text-sm hover:border-white/20 transition-colors"
                            >
                                <MapPin className="h-4 w-4 text-[#8fd6ff]" />
                                {currentCity.name}
                                <ChevronDown className={`h-4 w-4 transition-transform ${cityDropdownOpen ? 'rotate-180' : ''}`} />
                            </button>
                            {cityDropdownOpen && (
                                <div className="absolute top-full mt-1 right-0 bg-[#1b2328] border border-white/10 rounded-lg shadow-xl z-50 min-w-[160px]">
                                    {CITIES.map(city => (
                                        <button
                                            key={city.id}
                                            onClick={() => {
                                                setSelectedCity(city.id);
                                                setCityDropdownOpen(false);
                                            }}
                                            className={`w-full text-left px-4 py-2 text-sm hover:bg-white/5 transition-colors ${city.id === selectedCity ? 'text-[#8fd6ff]' : 'text-gray-300'
                                                } first:rounded-t-lg last:rounded-b-lg`}
                                        >
                                            {city.name}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>

                        <div className="flex items-center bg-[#1b2328] rounded-lg p-1 border border-white/10">
                            <button
                                className="px-3 py-1.5 rounded text-xs font-medium text-gray-400 hover:text-white"
                                onClick={() => router.push('/optimized')}
                            >
                                Single View
                            </button>
                            <button className="px-3 py-1.5 rounded text-xs font-medium bg-[#8fd6ff] text-[#101518]">
                                Split View
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex flex-1 overflow-hidden">
                {/* Left Panel - Metrics */}
                <aside
                    className="w-[420px] min-w-[420px] bg-[#0f1c23]/95 backdrop-blur-xl border-r border-white/10 flex flex-col overflow-hidden pointer-events-auto"
                >
                    <div className="flex-1 overflow-y-auto p-6 space-y-6 overscroll-contain">
                        {/* Mode Toggles */}
                        <div>
                            <h3 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                                <BarChart3 className="h-4 w-4" />
                                Active Modes
                            </h3>
                            <div className="flex flex-wrap gap-2">
                                {MODE_ORDER.map(modeId => {
                                    const mode = comparison?.modes[modeId];
                                    const isSelected = selectedModes.has(modeId);
                                    return (
                                        <button
                                            key={modeId}
                                            onClick={() => toggleMode(modeId)}
                                            disabled={!mode?.available}
                                            className={`px-3 py-2 rounded-lg text-xs font-medium transition-all border ${!mode?.available
                                                ? 'opacity-40 cursor-not-allowed border-gray-700 text-gray-500'
                                                : isSelected
                                                    ? 'border-white/30 text-white'
                                                    : 'border-white/10 text-gray-400 hover:border-white/20'
                                                }`}
                                            style={{
                                                backgroundColor: isSelected && mode?.available ? `${MODE_COLORS[modeId]}20` : 'transparent',
                                                borderColor: isSelected && mode?.available ? MODE_COLORS[modeId] : undefined,
                                            }}
                                        >
                                            <span
                                                className="inline-block w-2 h-2 rounded-full mr-2"
                                                style={{ backgroundColor: MODE_COLORS[modeId] }}
                                            />
                                            {mode?.name || modeId}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Mode Comparison Cards */}
                        {MODE_ORDER.map(modeId => {
                            const mode = comparison?.modes[modeId];
                            if (!mode?.available) return null;

                            return (
                                <div
                                    key={modeId}
                                    className={`p-4 rounded-xl border transition-all ${selectedModes.has(modeId)
                                        ? 'bg-[#1b2328]/80 border-white/20'
                                        : 'bg-[#1b2328]/30 border-white/5 opacity-60'
                                        }`}
                                    style={{
                                        borderLeftWidth: '3px',
                                        borderLeftColor: MODE_COLORS[modeId],
                                    }}
                                >
                                    <div className="flex items-center justify-between mb-4">
                                        <span className="text-sm font-medium text-white">{mode.name}</span>
                                        {renderImprovementBadge(mode.improvements.accessibility_mean)}
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <div className="text-xl font-bold text-white">
                                                {mode.metrics?.scores?.accessibility_mean?.toFixed(1) || '—'}
                                            </div>
                                            <div className="text-xs text-gray-500">Accessibility</div>
                                        </div>
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="text-xl font-bold text-white">
                                                    {mode.metrics?.scores?.travel_time_min_mean?.toFixed(1) || '—'}
                                                </span>
                                                {renderImprovementBadge(mode.improvements.travel_time_min_mean, true)}
                                            </div>
                                            <div className="text-xs text-gray-500">Avg Travel (min)</div>
                                        </div>
                                    </div>

                                    <div className="mt-3 pt-3 border-t border-white/10">
                                        <div className="text-xs text-gray-500">
                                            POIs: {allPois[modeId]?.length || 0} placements
                                        </div>
                                    </div>
                                </div>
                            );
                        })}

                        {error && (
                            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                                <p className="text-xs text-red-400">{error}</p>
                            </div>
                        )}

                        {/* No data message */}
                        {!loading && comparison && !Object.values(comparison.modes).some(m => m.available) && (
                            <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                                <p className="text-sm text-yellow-400">No optimization data available for {currentCity.name} yet.</p>
                                <p className="text-xs text-gray-400 mt-1">Run the optimization pipeline for this city first.</p>
                            </div>
                        )}
                    </div>
                </aside>

                {/* Map Area */}
                <div className="flex-1 relative pointer-events-auto">
                    <MapComponent
                        key={mapKey}
                        suggestions={combinedSuggestions}
                        selectedSuggestionIds={new Set()}
                        center={currentCity.center}
                        zoom={12}
                        className="absolute inset-0"
                    />

                    {/* Legend Overlay */}
                    <div className="absolute bottom-6 right-6 bg-[#0f1c23]/90 backdrop-blur-xl rounded-lg border border-white/10 p-3 z-10">
                        <div className="text-xs font-medium text-gray-400 mb-2">POI Layers</div>
                        {selectedModes.size === 0 ? (
                            <div className="text-xs text-gray-500 italic">Select a mode above</div>
                        ) : (
                            MODE_ORDER.map(modeId => {
                                const mode = comparison?.modes[modeId];
                                if (!mode?.available || !selectedModes.has(modeId)) return null;
                                return (
                                    <div key={modeId} className="flex items-center gap-2 text-xs text-gray-300">
                                        <span
                                            className="w-3 h-3 rounded-full"
                                            style={{ backgroundColor: MODE_COLORS[modeId] }}
                                        />
                                        {mode.name} ({allPois[modeId]?.length || 0})
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
