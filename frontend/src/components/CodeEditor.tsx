
import React, { useState } from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import TestCasePanel from "./TestCasePanel";
import OutputPanel from "./OutputPanel";
import { Play, Save, Code, Send } from "lucide-react";

interface CodeEditorProps {
  questionId: number;
  initialCode: string;
  onSubmit: (code: string, language: string) => void;
}

const languages = [
  { id: 'javascript', name: 'JavaScript' },
  { id: 'python', name: 'Python' },
  { id: 'java', name: 'Java' },
  { id: 'cpp', name: 'C++' },
  { id: 'golang', name: 'Go' }
];

const CodeEditor: React.FC<CodeEditorProps> = ({
  questionId,
  initialCode,
  onSubmit
}) => {
  const [code, setCode] = useState(initialCode);
  const [language, setLanguage] = useState('javascript');
  const [output, setOutput] = useState<null | { result: string; stdout: string; error: string | null }>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeTab, setActiveTab] = useState('editor');
  
  const handleRunCode = async (testCase?: string) => {
    try {
      setIsRunning(true);
      setOutput(null);
      
      // Simulate code execution delay
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Mock output - in a real app this would come from an API
      const mockOutput = {
        result: testCase ? `[1, 2, 3, 4, 5]` : `Success: Test cases passed (3/3)`,
        stdout: `Running test case...
Input: [1, 2, 3, 4]
Expected: [1, 2, 3, 4, 5]
Output: [1, 2, 3, 4, 5]
Test passed!`,
        error: null
      };
      
      setOutput(mockOutput);
      setActiveTab('output');
    } catch (error) {
      console.error('Error running code:', error);
      setOutput({
        result: 'Error',
        stdout: '',
        error: 'An error occurred while running the code.'
      });
    } finally {
      setIsRunning(false);
    }
  };
  
  const handleSubmitCode = async () => {
    try {
      setIsSubmitting(true);
      
      // Simulate submission delay
      await new Promise(resolve => setTimeout(resolve, 1200));
      
      // Call parent submit handler
      onSubmit(code, language);
      
      // Mock output
      setOutput({
        result: 'Success! All test cases passed.',
        stdout: 'Running all test cases...\nAll tests passed successfully!\nTime: 24ms, Memory: 8.5MB',
        error: null
      });
      setActiveTab('output');
    } catch (error) {
      console.error('Error submitting code:', error);
      setOutput({
        result: 'Error',
        stdout: '',
        error: 'An error occurred while submitting the code.'
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  const getLanguageTemplate = (lang: string) => {
    switch (lang) {
      case 'javascript':
        return `/**
 * @param {number[]} nums
 * @return {number[]}
 */
function solution(nums) {
    // Your code here
    return nums;
}`;
      case 'python':
        return `def solution(nums):
    # Your code here
    return nums`;
      case 'java':
        return `class Solution {
    public int[] solution(int[] nums) {
        // Your code here
        return nums;
    }
}`;
      case 'cpp':
        return `class Solution {
public:
    vector<int> solution(vector<int>& nums) {
        // Your code here
        return nums;
    }
};`;
      case 'golang':
        return `func solution(nums []int) []int {
    // Your code here
    return nums
}`;
      default:
        return '// Your code here';
    }
  };
  
  const handleLanguageChange = (value: string) => {
    if (value !== language) {
      // If code hasn't been modified or confirm the change
      if (code === initialCode || code === getLanguageTemplate(language) || 
          window.confirm('Changing language will reset your code. Continue?')) {
        setLanguage(value);
        setCode(getLanguageTemplate(value));
      }
    }
  };
  
  return (
    <div className="h-full flex flex-col bg-assessment-panel border-l border-assessment-border">
      <div className="px-4 py-3 border-b border-assessment-border flex items-center justify-between">
        <div className="flex items-center">
          <Code className="h-5 w-5 text-assessment-secondary mr-2" />
          <span className="font-medium text-assessment-secondary">Solution</span>
        </div>
        <div className="flex items-center space-x-2">
          <Select value={language} onValueChange={handleLanguageChange}>
            <SelectTrigger className="w-[140px] h-8">
              <SelectValue placeholder="Select language" />
            </SelectTrigger>
            <SelectContent>
              {languages.map(lang => (
                <SelectItem key={lang.id} value={lang.id}>
                  {lang.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <TabsList className="px-4 py-2 border-b border-assessment-border bg-assessment-sidebar justify-start">
          <TabsTrigger value="editor" className="text-sm">Editor</TabsTrigger>
          <TabsTrigger value="output" className="text-sm">Output</TabsTrigger>
        </TabsList>
        
        <TabsContent value="editor" className="flex-1 flex flex-col p-0 m-0">
          <ScrollArea className="flex-1">
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="code-editor w-full h-full p-4 font-mono text-sm"
              spellCheck="false"
            />
          </ScrollArea>
          
          <TestCasePanel onRunTestCase={handleRunCode} />
          
          <div className="p-4 border-t border-assessment-border flex items-center justify-between">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => handleRunCode()}
              disabled={isRunning || isSubmitting}
              className="flex items-center"
            >
              <Play className="mr-1 h-4 w-4" />
              Run
              {isRunning && <span className="ml-1">...</span>}
            </Button>
            
            <div className="flex items-center space-x-2">
              <Button 
                variant="outline" 
                size="sm"
                className="flex items-center"
              >
                <Save className="mr-1 h-4 w-4" />
                Save
              </Button>
              <Button 
                variant="default" 
                size="sm"
                onClick={handleSubmitCode}
                disabled={isRunning || isSubmitting}
                className="flex items-center bg-assessment-primary hover:bg-assessment-primary/90"
              >
                <Send className="mr-1 h-4 w-4" />
                Submit
                {isSubmitting && <span className="ml-1">...</span>}
              </Button>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="output" className="flex-1 flex flex-col p-0 m-0">
          {output ? (
            <OutputPanel output={output} />
          ) : (
            <div className="flex-1 flex items-center justify-center text-muted-foreground">
              Run your code to see output
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CodeEditor;
