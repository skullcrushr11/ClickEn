
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ChevronUp, ChevronDown, Play, X } from "lucide-react";

interface TestCasePanelProps {
  onRunTestCase: (testCase: string) => void;
}

const TestCasePanel: React.FC<TestCasePanelProps> = ({ onRunTestCase }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [testCase, setTestCase] = useState('[1, 2, 3, 4]');
  const [customTestCases, setCustomTestCases] = useState<string[]>([]);
  
  const handleAddTestCase = () => {
    if (testCase.trim() !== '') {
      setCustomTestCases(prev => [...prev, testCase]);
      setTestCase('');
    }
  };
  
  const handleRemoveTestCase = (index: number) => {
    setCustomTestCases(prev => prev.filter((_, i) => i !== index));
  };
  
  const handleRunTestCase = (tc: string) => {
    onRunTestCase(tc);
  };
  
  return (
    <div className={cn(
      "border-t border-assessment-border transition-all overflow-hidden",
      isExpanded ? "h-[180px]" : "h-[42px]"
    )}>
      <div className="p-2 flex items-center justify-between bg-assessment-sidebar">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center text-sm font-medium text-assessment-secondary"
        >
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 mr-1" />
          ) : (
            <ChevronDown className="h-4 w-4 mr-1" />
          )}
          Custom Test Cases
        </button>
      </div>
      
      {isExpanded && (
        <div className="p-3 bg-white flex flex-col space-y-3">
          <div className="flex items-center space-x-2">
            <div className="test-case-container flex-1">
              <textarea
                value={testCase}
                onChange={(e) => setTestCase(e.target.value)}
                placeholder="Enter test case input..."
                className="w-full h-[60px] p-2 text-sm font-mono resize-none outline-none"
              />
            </div>
            <div className="flex flex-col space-y-1">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleAddTestCase}
                className="text-xs h-7"
              >
                Add
              </Button>
              <Button 
                variant="default" 
                size="sm" 
                onClick={() => handleRunTestCase(testCase)}
                className="text-xs h-7 bg-assessment-primary hover:bg-assessment-primary/90"
              >
                <Play className="h-3 w-3 mr-1" />
                Run
              </Button>
            </div>
          </div>
          
          {customTestCases.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {customTestCases.map((tc, index) => (
                <div 
                  key={index}
                  className="flex items-center bg-assessment-sidebar rounded p-1 pr-2 text-xs"
                >
                  <button
                    onClick={() => handleRunTestCase(tc)}
                    className="mr-1 text-assessment-accent p-0.5 rounded hover:bg-assessment-primary/10"
                  >
                    <Play className="h-3 w-3" />
                  </button>
                  <span className="font-mono truncate max-w-[120px]">{tc}</span>
                  <button
                    onClick={() => handleRemoveTestCase(index)}
                    className="ml-1 text-assessment-secondary p-0.5 rounded hover:bg-assessment-primary/10"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TestCasePanel;
