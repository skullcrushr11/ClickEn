
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, ArrowRight, Clock } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const StudentDashboard: React.FC = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [upcomingTests, setUpcomingTests] = useState<any[]>([]);
  
  // Mock data for tests
  useEffect(() => {
    // In a real app, this would fetch from an API
    setUpcomingTests([
      {
        id: 'T001',
        title: 'Algorithm Assessment',
        description: 'Basic algorithm problems focusing on arrays and strings',
        date: '2023-07-15',
        duration: 60,
        questions: 5,
        status: 'scheduled'
      },
      {
        id: 'T002',
        title: 'Data Structures Quiz',
        description: 'Assessment on trees, graphs, and advanced data structures',
        date: '2023-07-20',
        duration: 90,
        questions: 7,
        status: 'scheduled'
      }
    ]);
  }, []);
  
  const handleStartTest = (testId: string) => {
    // In a real app, this would initialize a test session
    toast({
      title: "Test session started",
      description: "Your webcam and proctoring have been activated.",
    });
    navigate(`/test/${testId}`);
  };
  
  const handleLogout = () => {
    localStorage.removeItem('userType');
    localStorage.removeItem('userEmail');
    localStorage.removeItem('isAuthenticated');
    navigate('/');
  };
  
  return (
    <div className="min-h-screen bg-assessment-panel">
      <header className="bg-white border-b py-4 px-6">
        <div className="max-w-5xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">AssessQuest</h1>
          
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">student@example.com</span>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              Logout
            </Button>
          </div>
        </div>
      </header>
      
      <main className="max-w-5xl mx-auto py-8 px-6">
        <h2 className="text-2xl font-bold mb-6">Welcome to AssessQuest</h2>
        
        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Upcoming Assessments</CardTitle>
              <CardDescription>
                Your scheduled coding assessments and quizzes
              </CardDescription>
            </CardHeader>
            <CardContent>
              {upcomingTests.length > 0 ? (
                <div className="space-y-4">
                  {upcomingTests.map(test => (
                    <div 
                      key={test.id}
                      className="flex flex-col sm:flex-row sm:items-center justify-between p-4 border rounded-lg"
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-medium">{test.title}</h3>
                          <Badge>{test.questions} Questions</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          {test.description}
                        </p>
                        <div className="flex items-center mt-2 text-xs">
                          <Clock className="h-3.5 w-3.5 text-muted-foreground mr-1" />
                          <span className="text-muted-foreground">{test.duration} minutes</span>
                          <span className="mx-1">•</span>
                          <span className="text-muted-foreground">Date: {test.date}</span>
                        </div>
                      </div>
                      
                      <div className="mt-4 sm:mt-0 flex flex-col sm:items-end gap-2">
                        <Button 
                          className="flex items-center"
                          onClick={() => handleStartTest(test.id)}
                        >
                          Start Test
                          <ArrowRight className="ml-2 h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6">
                  <p className="text-muted-foreground">You have no upcoming assessments.</p>
                </div>
              )}
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="bg-yellow-50 border-b">
              <div className="flex items-start gap-2">
                <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
                <div>
                  <CardTitle>Important Instructions</CardTitle>
                  <CardDescription className="text-yellow-700">
                    Please read before starting any assessment
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-2 pt-6">
              <p className="text-sm">
                <span className="font-semibold">Proctoring:</span> All tests are proctored. We monitor your activity including keystrokes, mouse movements, and tab switching.
              </p>
              <p className="text-sm">
                <span className="font-semibold">Environment:</span> Make sure you're in a quiet place with a stable internet connection.
              </p>
              <p className="text-sm">
                <span className="font-semibold">Devices:</span> Use a laptop or desktop computer. Mobile devices are not supported.
              </p>
              <p className="text-sm">
                <span className="font-semibold">Browser:</span> Use Chrome, Firefox, or Edge for best experience.
              </p>
              <p className="text-sm">
                <span className="font-semibold">Cheating:</span> Copying code or switching tabs may flag you for cheating.
              </p>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default StudentDashboard;
