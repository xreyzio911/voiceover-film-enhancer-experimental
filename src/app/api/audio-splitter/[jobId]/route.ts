import { NextResponse, type NextRequest } from "next/server";
import { getServerAuthSession } from "@/auth";
import { isAllowedEmail } from "@/lib/authAllowlist";
import { getAudioSplitterJob } from "@/lib/audioSplitterJobs";
import { isLocalHost } from "@/lib/isLocalHost";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{ jobId: string }>;
};

export async function GET(request: NextRequest, context: RouteContext) {
  const host = request.headers.get("x-forwarded-host") ?? request.headers.get("host");
  const localMode = isLocalHost(host);
  const session = localMode ? null : await getServerAuthSession();
  if (!localMode && !isAllowedEmail(session?.user?.email)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { jobId } = await context.params;
  const job = getAudioSplitterJob(jobId);
  if (!job) {
    return NextResponse.json({ error: "Audio splitter job not found." }, { status: 404 });
  }

  return NextResponse.json(job, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}
