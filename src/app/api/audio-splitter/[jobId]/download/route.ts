import { NextResponse, type NextRequest } from "next/server";
import { getServerAuthSession } from "@/auth";
import { isAllowedEmail } from "@/lib/authAllowlist";
import { getAudioSplitterJob, getAudioSplitterJobDownload } from "@/lib/audioSplitterJobs";
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
  if (job.status !== "done") {
    return NextResponse.json({ error: "Audio splitter job is not complete yet." }, { status: 409 });
  }

  const download = getAudioSplitterJobDownload(jobId);
  if (!download) {
    return NextResponse.json({ error: "Audio splitter ZIP is unavailable." }, { status: 404 });
  }

  return new Response(new Uint8Array(download.zip), {
    status: 200,
    headers: {
      "Content-Type": "application/zip",
      "Content-Disposition": `attachment; filename="${download.zipName}"`,
      "Cache-Control": "no-store",
    },
  });
}
