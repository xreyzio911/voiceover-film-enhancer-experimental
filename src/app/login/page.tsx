import LoginCard from "@/components/LoginCard";
import { getServerAuthSession } from "@/auth";
import { isAllowedEmail } from "@/lib/authAllowlist";
import { isLocalHost } from "@/lib/isLocalHost";
import { headers } from "next/headers";
import { redirect } from "next/navigation";
import styles from "./page.module.css";

export default async function LoginPage() {
  const headerStore = await headers();
  const host = headerStore.get("x-forwarded-host") ?? headerStore.get("host");
  if (isLocalHost(host)) {
    redirect("/");
  }

  const session = await getServerAuthSession();
  const email = session?.user?.email?.toLowerCase();

  if (isAllowedEmail(email)) {
    redirect("/");
  }

  return (
    <div className={styles.page}>
      <div className={styles.shell}>
        <LoginCard />
      </div>
    </div>
  );
}
