
import React from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CheckCircle, AlertTriangle, Terminal } from "lucide-react";

interface OutputPanelProps {
  output: {
    result: string;
    stdout: string;
    error: string | null;
  };
}

const OutputPanel: React.FC<OutputPanelProps> = ({ output }) => {
  return (
    <div className="flex-1 flex flex-col">
      <div className="p-4 border-b border-assessment-border">
        <div className="flex items-start space-x-3">
          {output.error ? (
            <AlertTriangle className="h-5 w-5 text-assessment-error flex-shrink-0 mt-0.5" />
          ) : (
            <CheckCircle className="h-5 w-5 text-assessment-success flex-shrink-0 mt-0.5" />
          )}
          <div>
            <h3 className={`font-medium ${output.error ? 'text-assessment-error' : 'text-assessment-success'}`}>
              {output.error ? 'Execution Failed' : 'Execution Successful'}
            </h3>
            <p className="text-sm text-muted-foreground">{output.result}</p>
          </div>
        </div>
      </div>
      
      <Tabs defaultValue="stdout" className="flex-1 flex flex-col">
        <TabsList className="px-4 py-2 border-b border-assessment-border bg-assessment-sidebar justify-start">
          <TabsTrigger value="stdout" className="text-sm">Console Output</TabsTrigger>
          {output.error && <TabsTrigger value="error" className="text-sm">Error</TabsTrigger>}
        </TabsList>
        
        <TabsContent value="stdout" className="flex-1 p-0 m-0">
          <ScrollArea className="h-full">
            <pre className="p-4 text-sm font-mono whitespace-pre-wrap">
              {output.stdout || 'No output'}
            </pre>
          </ScrollArea>
        </TabsContent>
        
        {output.error && (
          <TabsContent value="error" className="flex-1 p-0 m-0">
            <ScrollArea className="h-full">
              <pre className="p-4 text-sm font-mono text-assessment-error whitespace-pre-wrap">
                {output.error}
              </pre>
            </ScrollArea>
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
};

export default OutputPanel;
