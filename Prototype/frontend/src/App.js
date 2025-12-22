import React, { useState, useEffect, useRef } from 'react';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';
import Header from './components/Header';
import ParameterConfigPanel from './components/ParameterConfigPanel';
import ActionControl from './components/ActionControl';
import LoggingPanel from './components/LoggingPanel';
import MapEmbedding from './components/MapEmbedding';
import { defaultParameters } from './config/parameters';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || '';

export const App = () => {
  const [parameters, setParameters] = useState(defaultParameters);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [showMap, setShowMap] = useState(false);
  const [mapTimestamp, setMapTimestamp] = useState(Date.now());
  const pollingIntervalRef = useRef(null);

  // Handle parameter changes
  const handleParameterChange = (category, paramKey, value) => {
    setParameters((prev) => ({
      ...prev,
      [category]: {
        ...prev[category],
        [paramKey]: {
          ...prev[category][paramKey],
          value,
        },
      },
    }));
  };

  // Start optimization
  const handleStartOptimization = async () => {
    try {
      setIsRunning(true);
      setLogs([]);
      setShowMap(false);

      // Start polling for logs
      startLogPolling();

      // Send optimization request
      const response = await axios.post(`${BACKEND_URL}/api/run-optimization`, {
        parameters,
      });

      if (response.data.status === 'started') {
        toast.info('Optimization started. Please wait...');
      }
    } catch (error) {
      console.error('Optimization error:', error);
      toast.error('Failed to start optimization. Check backend connection.');
      setIsRunning(false);
      stopLogPolling();
    }
  };

  // Start log polling
  const startLogPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    pollingIntervalRef.current = setInterval(async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/api/logs/latest`);
        const { logs: newLogs, running } = response.data;
        
        setLogs(newLogs);
        
        // Check if optimization is complete
        if (!running && newLogs.length > 0) {
          const lastLog = newLogs[newLogs.length - 1];
          if (lastLog.message && lastLog.message.includes('Optimization completed successfully')) {
            toast.success('Optimization completed successfully!');
            setShowMap(true);
            setMapTimestamp(Date.now());
            setIsRunning(false);
            stopLogPolling();
          }
        }
      } catch (error) {
        console.error('Error fetching logs:', error);
      }
    }, 500); // Poll every 500ms
  };

  // Stop log polling
  const stopLogPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopLogPolling();
    };
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Toaster position="top-right" richColors />
      
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Parameter Configuration */}
        <section className="mb-8">
          <ParameterConfigPanel
            parameters={parameters}
            onParameterChange={handleParameterChange}
            disabled={isRunning}
          />
        </section>

        {/* Action Control */}
        <section className="mb-8">
          <ActionControl
            onStart={handleStartOptimization}
            disabled={isRunning}
            isRunning={isRunning}
          />
        </section>

        {/* Logging Panel */}
        {logs.length > 0 && (
          <section className="mb-8">
            <LoggingPanel logs={logs} />
          </section>
        )}

        {/* Map Embedding */}
        {showMap && (
          <section className="mb-8">
            <MapEmbedding timestamp={mapTimestamp} />
          </section>
        )}
      </main>
    </div>
  );
};

export default App;
