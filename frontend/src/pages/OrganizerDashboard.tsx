
import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useNavigate } from 'react-router-dom';
import QuestionForm from '@/components/organizer/QuestionForm';
import TestSessionConfig from '@/components/organizer/TestSessionConfig';
import CheatDetectionPanel from '@/components/organizer/CheatDetectionPanel';
import { CheckCircle, Plus, Users, AlertTriangle, Clock } from 'lucide-react';

const OrganizerDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');

  // Mock data - in a real app this would come from an API
  const testSessions = [
    {
      id: 'T001',
      title: 'Algorithm Assessment',
      status: 'active',
      startDate: '2023-07-10',
      endDate: '2023-07-15',
      totalCandidates: 25,
      completedCandidates: 12,
      flaggedCandidates: 3
    },
    {
      id: 'T002',
      title: 'Data Structures Quiz',
      status: 'active',
      startDate: '2023-07-15',
      endDate: '2023-07-20',
      totalCandidates: 18,
      completedCandidates: 0,
      flaggedCandidates: 0
    },
    {
      id: 'T003',
      title: 'Web Development Test',
      status: 'completed',
      startDate: '2023-06-20',
      endDate: '2023-06-25',
      totalCandidates: 30,
      completedCandidates: 28,
      flaggedCandidates: 5
    }
  ];

  const handleLogout = () => {
    localStorage.removeItem('userType');
    localStorage.removeItem('userEmail');
    localStorage.removeItem('isAuthenticated');
    navigate('/');
  };

  const handleAddQuestion = (data: any) => {
    console.log('New question data:', data);
    setActiveTab('overview');
    // In a real app, this would save the data to a database
  };

  const handleCreateSession = (data: any) => {
    console.log('New test session data:', data);
    setActiveTab('overview');
    // In a real app, this would save the data to a database
  };

  return (
    <div className="min-h-screen bg-assessment-panel">
      <header className="bg-white border-b py-4 px-6">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">ClickEn Admin</h1>

          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">admin@example.com</span>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-8 px-6">
        <Tabs
          defaultValue="overview"
          value={activeTab}
          onValueChange={setActiveTab}
          className="w-full"
        >
          <div className="flex justify-between items-center mb-6">
            <TabsList className="grid w-[600px] grid-cols-4">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="add-question">Add Question</TabsTrigger>
              <TabsTrigger value="test-session">Test Session</TabsTrigger>
              <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Total Sessions
                  </CardTitle>
                  <Clock className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{testSessions.length}</div>
                  <p className="text-xs text-muted-foreground">
                    {testSessions.filter(s => s.status === 'active').length} active sessions
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Total Candidates
                  </CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {testSessions.reduce((acc, session) => acc + session.totalCandidates, 0)}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {testSessions.reduce((acc, session) => acc + session.completedCandidates, 0)} completed tests
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Flagged for Cheating
                  </CardTitle>
                  <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {testSessions.reduce((acc, session) => acc + session.flaggedCandidates, 0)}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Across all test sessions
                  </p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Test Sessions</CardTitle>
                <CardDescription>
                  Manage your active and past assessment sessions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {testSessions.map(session => (
                    <div
                      key={session.id}
                      className="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div>
                        <h3 className="font-medium">{session.title}</h3>
                        <div className="text-sm text-muted-foreground">
                          {session.startDate} to {session.endDate}
                        </div>
                        <div className="flex items-center mt-1 text-xs">
                          <span className="text-muted-foreground">{session.totalCandidates} candidates</span>
                          <span className="mx-1">•</span>
                          <span className="text-green-600">{session.completedCandidates} completed</span>
                          {session.flaggedCandidates > 0 && (
                            <>
                              <span className="mx-1">•</span>
                              <span className="text-red-600">{session.flaggedCandidates} flagged</span>
                            </>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {session.status === 'active' ? (
                          <Button variant="outline" size="sm">
                            Manage
                          </Button>
                        ) : (
                          <Button variant="ghost" size="sm">
                            View Results
                          </Button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
              <CardFooter className="border-t px-6 py-4">
                <Button
                  className="w-full"
                  onClick={() => setActiveTab('test-session')}
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Create New Test Session
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="add-question">
            <QuestionForm onSubmit={handleAddQuestion} />
          </TabsContent>

          <TabsContent value="test-session">
            <TestSessionConfig onSubmit={handleCreateSession} />
          </TabsContent>

          <TabsContent value="monitoring">
            <CheatDetectionPanel />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default OrganizerDashboard;
