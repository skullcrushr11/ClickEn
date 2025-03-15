
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/hooks/use-toast";
import { Plus, Trash2 } from "lucide-react";

const questionSchema = z.object({
  title: z.string().min(5, { message: "Question title must be at least 5 characters." }),
  description: z.string().min(20, { message: "Question description must be at least 20 characters." }),
  difficulty: z.enum(["easy", "medium", "hard"]),
  timeLimit: z.string().min(1, { message: "Time limit is required" }),
  constraints: z.string().min(1, { message: "At least one constraint is required" }),
  tags: z.string(),
  timeComplexity: z.string().optional(),
  spaceComplexity: z.string().optional(),
  starterCodeJs: z.string(),
  starterCodePython: z.string(),
  starterCodeJava: z.string(),
  starterCodeCpp: z.string(),
  starterCodeGolang: z.string(),
});

type QuestionFormValues = z.infer<typeof questionSchema>;

interface QuestionFormProps {
  onSubmit: (data: any) => void;
}

const QuestionForm: React.FC<QuestionFormProps> = ({ onSubmit }) => {
  const { toast } = useToast();
  const [examples, setExamples] = useState([
    { input: '', output: '', explanation: '' }
  ]);
  
  const form = useForm<QuestionFormValues>({
    resolver: zodResolver(questionSchema),
    defaultValues: {
      title: "",
      description: "",
      difficulty: "medium",
      timeLimit: "30",
      constraints: "",
      tags: "arrays,algorithms",
      timeComplexity: "O(n)",
      spaceComplexity: "O(1)",
      starterCodeJs: "/**\n * @param {number[]} nums\n * @return {number[]}\n */\nfunction solution(nums) {\n    // Your code here\n    return nums;\n}",
      starterCodePython: "def solution(nums):\n    # Your code here\n    return nums",
      starterCodeJava: "class Solution {\n    public int[] solution(int[] nums) {\n        // Your code here\n        return nums;\n    }\n}",
      starterCodeCpp: "class Solution {\npublic:\n    vector<int> solution(vector<int>& nums) {\n        // Your code here\n        return nums;\n    }\n};",
      starterCodeGolang: "func solution(nums []int) []int {\n    // Your code here\n    return nums\n}",
    },
  });

  const handleSubmit = (data: QuestionFormValues) => {
    const formattedData = {
      ...data,
      examples: examples,
      tags: data.tags.split(',').map(tag => tag.trim()),
      constraints: data.constraints.split('\n').filter(c => c.trim().length > 0),
    };
    
    onSubmit(formattedData);
    
    toast({
      title: "Question created",
      description: "Your question has been successfully added to the assessment.",
    });
  };

  const addExample = () => {
    setExamples([...examples, { input: '', output: '', explanation: '' }]);
  };

  const removeExample = (index: number) => {
    setExamples(examples.filter((_, i) => i !== index));
  };

  const updateExample = (index: number, field: string, value: string) => {
    const updatedExamples = [...examples];
    updatedExamples[index] = { ...updatedExamples[index], [field]: value };
    setExamples(updatedExamples);
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-6">Create New Question</h2>
      
      <Form {...form}>
        <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <FormField
              control={form.control}
              name="title"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Question Title</FormLabel>
                  <FormControl>
                    <Input placeholder="Two Sum" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="difficulty"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Difficulty</FormLabel>
                    <Select 
                      onValueChange={field.onChange} 
                      defaultValue={field.value}
                    >
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select difficulty" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="easy">Easy</SelectItem>
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="hard">Hard</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              
              <FormField
                control={form.control}
                name="timeLimit"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Time Limit (minutes)</FormLabel>
                    <FormControl>
                      <Input type="number" min="1" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          </div>
          
          <FormField
            control={form.control}
            name="description"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Question Description</FormLabel>
                <FormControl>
                  <Textarea 
                    rows={5}
                    placeholder="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target..." 
                    {...field} 
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">Examples</h3>
              <Button 
                type="button" 
                variant="outline" 
                size="sm" 
                onClick={addExample}
                className="text-xs"
              >
                <Plus className="h-4 w-4 mr-1" /> Add Example
              </Button>
            </div>
            
            {examples.map((example, index) => (
              <div key={index} className="p-4 bg-assessment-sidebar rounded-md space-y-4 relative">
                <div className="absolute top-2 right-2">
                  {index > 0 && (
                    <Button 
                      type="button" 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => removeExample(index)}
                      className="h-8 w-8 p-0"
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  )}
                </div>
                
                <div>
                  <FormLabel className="text-xs">Input</FormLabel>
                  <Textarea 
                    rows={2}
                    value={example.input}
                    onChange={(e) => updateExample(index, 'input', e.target.value)}
                    placeholder="[2, 7, 11, 15], target = 9"
                  />
                </div>
                
                <div>
                  <FormLabel className="text-xs">Output</FormLabel>
                  <Textarea 
                    rows={2}
                    value={example.output}
                    onChange={(e) => updateExample(index, 'output', e.target.value)}
                    placeholder="[0, 1]"
                  />
                </div>
                
                <div>
                  <FormLabel className="text-xs">Explanation (Optional)</FormLabel>
                  <Textarea 
                    rows={2}
                    value={example.explanation}
                    onChange={(e) => updateExample(index, 'explanation', e.target.value)}
                    placeholder="Because nums[0] + nums[1] = 2 + 7 = 9, we return [0, 1]."
                  />
                </div>
              </div>
            ))}
          </div>
          
          <FormField
            control={form.control}
            name="constraints"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Constraints (one per line)</FormLabel>
                <FormControl>
                  <Textarea 
                    rows={3}
                    placeholder="2 <= nums.length <= 10^4\n-10^9 <= nums[i] <= 10^9\n-10^9 <= target <= 10^9" 
                    {...field} 
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <FormField
              control={form.control}
              name="tags"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Tags (comma separated)</FormLabel>
                  <FormControl>
                    <Input placeholder="arrays, hash-table" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <FormField
              control={form.control}
              name="timeComplexity"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Expected Time Complexity</FormLabel>
                  <FormControl>
                    <Input placeholder="O(n)" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <FormField
              control={form.control}
              name="spaceComplexity"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Expected Space Complexity</FormLabel>
                  <FormControl>
                    <Input placeholder="O(1)" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Starter Code</h3>
            
            <FormField
              control={form.control}
              name="starterCodeJs"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>JavaScript</FormLabel>
                  <FormControl>
                    <Textarea rows={6} {...field} className="font-mono text-sm" />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <FormField
              control={form.control}
              name="starterCodePython"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Python</FormLabel>
                  <FormControl>
                    <Textarea rows={6} {...field} className="font-mono text-sm" />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <FormField
                control={form.control}
                name="starterCodeJava"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Java</FormLabel>
                    <FormControl>
                      <Textarea rows={6} {...field} className="font-mono text-sm" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              
              <FormField
                control={form.control}
                name="starterCodeCpp"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>C++</FormLabel>
                    <FormControl>
                      <Textarea rows={6} {...field} className="font-mono text-sm" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              
              <FormField
                control={form.control}
                name="starterCodeGolang"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Go</FormLabel>
                    <FormControl>
                      <Textarea rows={6} {...field} className="font-mono text-sm" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          </div>
          
          <Button type="submit" className="w-full">
            Create Question
          </Button>
        </form>
      </Form>
    </div>
  );
};

export default QuestionForm;
