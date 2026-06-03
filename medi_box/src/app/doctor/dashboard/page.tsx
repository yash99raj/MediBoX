"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Avatar } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { toast } from "sonner"
import { Loader2 } from "lucide-react"
import { DiagnosisData } from "@/lib/store/diagnosesStore"

interface PendingDiagnosis {
  id: string;
  patientName: string;
  dateSubmitted: string;
  type: string;
  aiDiagnosis: string;
  status: string;
}

export default function DoctorDashboard() {
  const [pendingDiagnoses, setPendingDiagnoses] = useState<PendingDiagnosis[]>([])
  const [loading, setLoading] = useState(true)
  const [approvedCount, setApprovedCount] = useState(42)

  const fetchPendingDiagnoses = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/diagnoses?status=pending')
      if (response.ok) {
        const data = await response.json()
        
        // Map backend store data to frontend dashboard fields
        const formatted = data.diagnoses.map((d: DiagnosisData) => {
          let name = "John Doe"
          if (d.id === "1") name = "John Doe"
          else if (d.id === "2") name = "Jane Smith"
          else if (d.id === "3") name = "Robert Johnson"
          else name = `Patient #${d.id}`
          
          return {
            id: d.id,
            patientName: name,
            dateSubmitted: d.diagnosisDate,
            type: d.type,
            aiDiagnosis: `${d.aiDiagnosis} (${d.confidence}% confidence)`,
            status: d.status
          }
        })
        setPendingDiagnoses(formatted)
      } else {
        toast.error("Failed to fetch pending requests")
      }
    } catch (err) {
      console.error(err)
      toast.error("Failed to fetch pending requests")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPendingDiagnoses()
  }, [])

  const handleApprove = async (id: string) => {
    const feedback = prompt("Enter your medical opinion / prescription notes:", "AI diagnosis approved. Undergo general observation.");
    if (feedback === null) return;
    if (!feedback.trim()) {
      toast.error("Doctor feedback is required to approve the diagnosis.")
      return
    }

    try {
      const response = await fetch(`/api/diagnoses/${id}/review`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'approve',
          feedback,
          doctorName: "Dr. Sarah Williams",
        }),
      })

      if (response.ok) {
        toast.success(`Diagnosis #${id} approved successfully`)
        setApprovedCount(prev => prev + 1)
        fetchPendingDiagnoses()
      } else {
        toast.error("Failed to approve diagnosis.")
      }
    } catch {
      toast.error("An error occurred during approval.")
    }
  }

  const handleReject = async (id: string) => {
    const feedback = prompt("Enter rejection reason and notes:");
    if (feedback === null) return;
    if (!feedback.trim()) {
      toast.error("Doctor feedback is required to reject the diagnosis.")
      return
    }

    try {
      const response = await fetch(`/api/diagnoses/${id}/review`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'reject',
          feedback,
          doctorName: "Dr. Sarah Williams",
        }),
      })

      if (response.ok) {
        toast.success(`Diagnosis #${id} rejected successfully`)
        fetchPendingDiagnoses()
      } else {
        toast.error("Failed to reject diagnosis.")
      }
    } catch {
      toast.error("An error occurred during rejection.")
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 p-4 md:p-8">
      <header className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Avatar className="h-12 w-12 border-2 border-primary">
              <div className="flex h-full w-full items-center justify-center rounded-full bg-muted">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-6 w-6">
                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                  <circle cx="12" cy="7" r="4"></circle>
                </svg>
              </div>
            </Avatar>
            <div>
              <h1 className="text-3xl font-bold">Doctor Dashboard</h1>
              <p className="text-muted-foreground">Dr. Sarah Williams, MD</p>
            </div>
          </div>
          <Link href="/">
            <Button variant="outline">Logout</Button>
          </Link>
        </div>
      </header>

      <main className="max-w-7xl mx-auto space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">Pending Diagnoses</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-3xl font-bold">-</div>
              ) : (
                <div className="text-3xl font-bold">{pendingDiagnoses.length}</div>
              )}
              <p className="text-sm text-muted-foreground">Waiting for your review</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">Approved Diagnoses</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{approvedCount}</div>
              <p className="text-sm text-muted-foreground">This month</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">AI Accuracy</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">94%</div>
              <p className="text-sm text-muted-foreground">Based on your feedback</p>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Pending Diagnosis Requests</CardTitle>
            <CardDescription>
              Review AI-generated diagnoses and provide your medical opinion
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="h-8 w-8 text-primary animate-spin" />
              </div>
            ) : pendingDiagnoses.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Patient</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>AI Diagnosis</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pendingDiagnoses.map((diagnosis) => (
                    <TableRow key={diagnosis.id}>
                      <TableCell className="font-medium">{diagnosis.patientName}</TableCell>
                      <TableCell>{diagnosis.dateSubmitted}</TableCell>
                      <TableCell>{diagnosis.type}</TableCell>
                      <TableCell>{diagnosis.aiDiagnosis}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-800/20 dark:text-yellow-500">
                          Pending
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-2">
                          <Link href={`/doctor/diagnosis/${diagnosis.id}`}>
                            <Button size="sm" variant="outline">View</Button>
                          </Link>
                          <Button 
                            size="sm" 
                            variant="default"
                            onClick={() => handleApprove(diagnosis.id)}
                          >
                            Approve
                          </Button>
                          <Button 
                            size="sm" 
                            variant="destructive"
                            onClick={() => handleReject(diagnosis.id)}
                          >
                            Reject
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p>No pending diagnoses to review.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
} 