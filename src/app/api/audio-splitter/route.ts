import { NextResponse, type NextRequest } from "next/server";
import { getServerAuthSession } from "@/auth";
import { isAllowedEmail } from "@/lib/authAllowlist";
import { startAudioSplitterJob } from "@/lib/audioSplitterJobs";
import { isLocalHost } from "@/lib/isLocalHost";
import type { AudioSplitterInput } from "@/lib/audioSplitterService";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const isUploadFile = (value: FormDataEntryValue): value is File =>
  typeof value === "object" &&
  value !== null &&
  "arrayBuffer" in value &&
  "name" in value &&
  typeof value.arrayBuffer === "function";

export async function POST(request: NextRequest) {
  const host = request.headers.get("x-forwarded-host") ?? request.headers.get("host");
  const localMode = isLocalHost(host);
  const session = localMode ? null : await getServerAuthSession();
  if (!localMode && !isAllowedEmail(session?.user?.email)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json({ error: "Expected multipart/form-data with one or more files." }, { status: 400 });
  }

  const uploads = formData.getAll("files").filter(isUploadFile);
  if (uploads.length === 0) {
    return NextResponse.json({ error: "Upload at least one .wav file." }, { status: 400 });
  }

  try {
    const inputs: AudioSplitterInput[] = await Promise.all(
      uploads.map(async (file) => ({
        originalName: file.name,
        bytes: new Uint8Array(await file.arrayBuffer()),
        mimeType: file.type,
      })),
    );
    const job = startAudioSplitterJob(inputs);

    return NextResponse.json(job, {
      status: 202,
      headers: {
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 },
    );
  }
}
