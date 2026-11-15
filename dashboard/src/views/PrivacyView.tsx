import type { ReactNode } from "react";
import { Card, SectionTitle } from "../components/ui";

const REPO_URL = "https://github.com/arvindkushwaha55/PREDICAI";

function Block({
  heading,
  children,
}: {
  heading: string;
  children: ReactNode;
}) {
  return (
    <section>
      <h3 className="font-mono text-xs font-semibold tracking-[0.14em] text-slate-900 uppercase dark:text-slate-100">
        {heading}
      </h3>
      <div className="mt-1.5 space-y-2 text-sm leading-relaxed text-slate-700 dark:text-slate-300">
        {children}
      </div>
    </section>
  );
}

export function PrivacyView() {
  return (
    <div className="max-w-3xl space-y-6">
      <Card className="space-y-5 p-5" ticks>
        <div>
          <SectionTitle
            title="Privacy Policy"
            source="How this site handles information about its visitors."
          />
          <p className="mono text-xs text-slate-500 dark:text-slate-400">
            Last updated: 15 November 2026
          </p>
        </div>

        <Block heading="What this site is">
          <p>
            PREDICAI is a static research dashboard. It presents results
            from a predictive maintenance study built on public benchmark
            datasets. The site consists of static files (HTML, CSS,
            JavaScript and bundled data files) served to your browser; it
            has no backend of its own.
          </p>
        </Block>

        <Block heading="Information we collect">
          <p>
            None. The site has no user accounts, no sign-up, no contact
            form and no comment system. It does not ask for, receive or
            store personal information. The site sets no cookies of its own
            and runs no analytics, advertising or tracking scripts. Your
            theme preference (light or dark) is stored in your own
            browser's local storage so the site can remember it; it never
            leaves your device.
          </p>
        </Block>

        <Block heading="Data shown on the site">
          <p>
            Every figure on the site is bundled with the page when it
            loads. The fleet, model and platform pages display numbers
            exported from public research datasets and from documented
            experiment records in the project repository. The Maintenance
            page shows demo service rows that are synthetic, and the page
            itself labels them as synthetic. Nothing you do on the site is
            recorded by the site.
          </p>
        </Block>

        <Block heading="Hosting on GitHub Pages">
          <p>
            The site is hosted on GitHub Pages, a service operated by
            GitHub, Inc. When you visit, GitHub's servers may record
            standard technical information such as your IP address, your
            browser type and the pages requested, as described in the
            GitHub Privacy Statement. That processing is carried out by
            GitHub, not by this project, and this project does not receive
            or use those logs.
          </p>
        </Block>

        <Block heading="The Live API panel">
          <p>
            One panel (Model Info &amp; Live API) can check a small
            prediction API. In this hosted build it only attempts to
            contact an API running on your own computer. No request is
            sent to any third-party server by that panel. If no local
            API is running, the panel reports it as offline and shows
            no substitute numbers.
          </p>
        </Block>

        <Block heading="External links">
          <p>
            The site links to the project repository on GitHub. When you
            follow a link to GitHub, GitHub's own privacy statement
            applies to your visit there.
          </p>
        </Block>

        <Block heading="Contact">
          <p>
            Questions about this site or this policy can be raised through
            the project repository on GitHub:{" "}
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="font-medium text-cyan-700 underline-offset-2 hover:underline dark:text-cyan-300"
            >
              arvindkushwaha55/PREDICAI
            </a>
            .
          </p>
        </Block>

        <Block heading="Changes to this policy">
          <p>
            If this policy changes, the revised version will be published
            on this page with a new "last updated" date.
          </p>
        </Block>
      </Card>
    </div>
  );
}
