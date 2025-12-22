import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Terminal } from 'lucide-react';

const getLogLevelClass = (level) => {
  switch (level?.toLowerCase()) {
    case 'error':
    case 'critical':
      return 'text-[hsl(var(--log-error))]';
    case 'warning':
      return 'text-[hsl(var(--log-warning))]';
    case 'info':
    default:
      return 'text-[hsl(var(--log-info))]';
  }
};

export const LoggingPanel = ({ logs }) => {
  const logContainerRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const handleScroll = () => {
    if (logContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 50;
      setAutoScroll(isNearBottom);
    }
  };

  return (
    <Card className="border-border">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent">
            <Terminal className="h-5 w-5" />
          </div>
          <div>
            <CardTitle className="text-lg">Optimization Logs</CardTitle>
            <CardDescription className="text-xs">
              Real-time output from the backend optimization process
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div
          ref={logContainerRef}
          onScroll={handleScroll}
          className="log-panel h-[280px] overflow-y-auto rounded-lg bg-[hsl(var(--log-bg))] p-4 font-mono text-sm"
        >
          {logs.length === 0 ? (
            <div className="flex h-full items-center justify-center text-[hsl(var(--log-text))] opacity-50">
              Waiting for logs...
            </div>
          ) : (
            <div className="space-y-1">
              {logs.map((log, index) => (
                <div key={index} className="flex gap-2 text-[hsl(var(--log-text))]">
                  <span className="opacity-50">{log.timestamp || new Date().toLocaleTimeString()}</span>
                  <span className={getLogLevelClass(log.level)}>
                    [{log.level || 'INFO'}]
                  </span>
                  <span className="flex-1">{log.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        {!autoScroll && (
          <div className="mt-2 text-center">
            <button
              onClick={() => {
                setAutoScroll(true);
                if (logContainerRef.current) {
                  logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
                }
              }}
              className="text-xs text-accent hover:underline"
            >
              Resume auto-scroll
            </button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default LoggingPanel;
