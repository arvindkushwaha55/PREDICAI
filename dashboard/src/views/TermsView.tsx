import type { ReactNode } from "react";
import { Card, SectionTitle } from "../components/ui";

const REPO_URL = "https://github.com/arvindkushwaha55/PREDICAI";
const LICENCE_URL =
  "https://github.com/arvindkushwaha55/PREDICAI/blob/main/LICENSE";

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

export function TermsView() {
  return (
    <div className="max-w-3xl space-y-6">
      <Card className="space-y-5 p-5" ticks>
        <div>
          <SectionTitle
            title="Terms and Conditions"
            source="The terms that govern use of this site and its contents."
          />
          <p className="mono text-xs text-slate-500 dark:text-slate-400">
            Last updated: 15 November 2026
          </p>
        </div>

        <Block heading="Purpose of the site">
          <p>
            PREDICAI is a research and education demonstration of
            predictive maintenance methods: remaining useful life (RUL)
            estimation and fault classification, developed and evaluated on
            public benchmark datasets. It is a portfolio and study project,
            not a commercial product or service.
          </p>
        </Block>

        <Block heading="Not for operational use">
          <p>
            Model outputs shown on this site were computed on public
            benchmark data and are documented in experiment records in the
            project repository. They have not been validated on real
            machinery. You must not use this site, its predictions, its
            RUL values or its policy thresholds to make safety,
            maintenance or operational decisions about real equipment. The
            status bands shown in the dashboard (Monitor, Inspection,
            Review) are planning prompts defined by a demonstration
            policy; they are not validated failure boundaries.
          </p>
        </Block>

        <Block heading="Code licence">
          <p>
            The source code of this project is released under the MIT
            License. The full licence text is in the{" "}
            <a
              href={LICENCE_URL}
              target="_blank"
              rel="noreferrer"
              className="font-medium text-cyan-700 underline-offset-2 hover:underline dark:text-cyan-300"
            >
              LICENSE file in the repository
            </a>
            .
          </p>
        </Block>

        <Block heading="Dataset licences">
          <p>
            The datasets presented on this site remain the property of
            their providers and are used under their own terms. Nothing on
            this site relicenses them. The principal datasets are: NASA
            C-MAPSS and the NASA Li-ion Battery Aging dataset (NASA Ames
            Prognostics Center of Excellence), the CWRU Bearing dataset
            (Case Western Reserve University), the Paderborn Bearing
            dataset (Paderborn University, CC BY-NC 4.0), AI4I 2020 and
            SECOM (UCI Machine Learning Repository), and the IMS Bearing
            dataset (University of Cincinnati, distributed by NASA PCoE).
            The Datasets page records the licence status of every dataset
            in the registry, including the ones whose licence is not
            stated by the source.
          </p>
        </Block>

        <Block heading="Accuracy and availability">
          <p>
            Content is provided "as is", without warranties of any kind,
            express or implied. Figures are reported as measured in the
            documented experiment records, including negative results; no
            guarantee is given that any figure generalises beyond the
            protocol recorded with it. The maintainer may update, correct
            or remove any part of the site at any time.
          </p>
        </Block>

        <Block heading="Changes to these terms">
          <p>
            These terms may be revised. The current version is the one
            published on this page, and revisions take effect when they
            are published here with a new "last updated" date. Questions
            about these terms can be raised through the{" "}
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="font-medium text-cyan-700 underline-offset-2 hover:underline dark:text-cyan-300"
            >
              project repository on GitHub
            </a>
            .
          </p>
        </Block>
      </Card>
    </div>
  );
}
