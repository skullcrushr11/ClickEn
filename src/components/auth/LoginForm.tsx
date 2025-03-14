
import React from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from 'react-router-dom';

const loginSchema = z.object({
  email: z.string().email({ message: "Please enter a valid email address." }),
  password: z.string().min(6, { message: "Password must be at least 6 characters." }),
});

type LoginFormValues = z.infer<typeof loginSchema>;

interface LoginFormProps {
  userType: 'student' | 'organizer';
}

const LoginForm: React.FC<LoginFormProps> = ({ userType }) => {
  const { toast } = useToast();
  const navigate = useNavigate();
  
  const form = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const onSubmit = (data: LoginFormValues) => {
    // In a real app, this would connect to a backend service
    console.log("Login data:", data);
    
    // Mock authentication
    if (data.email && data.password) {
      localStorage.setItem('userType', userType);
      localStorage.setItem('userEmail', data.email);
      localStorage.setItem('isAuthenticated', 'true');
      
      toast({
        title: "Login successful",
        description: `Logged in as ${userType}`,
      });
      
      if (userType === 'student') {
        navigate('/student-dashboard');
      } else {
        navigate('/organizer-dashboard');
      }
    }
  };

  return (
    <div className="w-full max-w-md mx-auto p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold text-center mb-6">{userType === 'student' ? 'Student Login' : 'Organizer Login'}</h2>
      
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Email</FormLabel>
                <FormControl>
                  <Input placeholder="your.email@example.com" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <FormField
            control={form.control}
            name="password"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Password</FormLabel>
                <FormControl>
                  <Input type="password" placeholder="******" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <Button type="submit" className="w-full">
            Log in
          </Button>
        </form>
      </Form>
      
      <div className="mt-4 text-center text-sm">
        <p className="text-muted-foreground">
          {userType === 'student' 
            ? "Taking an assessment? Log in with your credentials above."
            : "Assessment organizer? Log in to manage assessments and candidates."}
        </p>
      </div>
    </div>
  );
};

export default LoginForm;
