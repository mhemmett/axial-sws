/* Prose content for the Axial Seamount shear-wave-splitting site.
   Plain data only — no imports, no build step. Consumed by index.html. */

window.AXIAL_CONTENT = {

  site: {
    title: "Axial SWS",
    tagline: "Shear-wave splitting and crack-induced anisotropy at Axial Seamount, across the April 2015 eruption and the inflation cycle that followed.",
    updated: "2026-09-29"
  },

  tabs: {

    overview: {
      label: "Overview",
      lead: "<p>Shear-wave splitting analysis of local seismicity at Axial Seamount, Juan de Fuca Ridge, using ocean-bottom seismometer data from the OOI Regional Cabled Array. The object is the temporal evolution of stress and crack-induced anisotropy across the <strong>April 2015 eruption</strong> and the long re-inflation that has followed it.</p>",
      body: [
        "<p>A shear wave crossing an anisotropic medium splits into two orthogonally polarised waves travelling at different speeds. The two parameters recovered from that splitting — the fast polarisation direction φ and the delay time δt between the fast and slow arrivals — measure the orientation and the density of aligned, fluid-filled cracks in the shear-wave window. Under the extensive-dilatancy interpretation, φ tracks the orientation of the maximum horizontal compressive stress holding those cracks open, and δt integrates crack density along the raypath. Tracking φ and δt in space and time is therefore a way to watch a volcanic stress field reorganise.</p>",

        "<p>Axial Seamount is an unusually good place to do this. It is the most magmatically active submarine volcano on the Juan de Fuca Ridge, it erupted in April 2015 while fully instrumented, and the OOI Regional Cabled Array has recorded it continuously since — a permanent, cabled, six-station caldera network with a co-located BOTPT geodetic record of seafloor uplift. The volcano has been re-inflating ever since the 2015 eruption drained it, recovering roughly 2.6 m of Central Caldera uplift to date. That gives an unbroken observational arc: a pre-eruption inflated state, the eruption itself, post-eruption deflation, and a decade of re-inflation, all sampled by tens of thousands of local earthquakes recorded on the same instruments.</p>",

        "<p>The analysis now covers two eras — <strong>2015–2021</strong> and <strong>2022–2026</strong> — at all six caldera stations, processed through a single measurement chain so that the two halves are directly comparable. The comparison of interest is not any one rose plot but the difference between them: whether φ rotates as the caldera re-pressurises, whether δt grows with crack density, and whether either tracks the geodetically measured uplift.</p>",

        "<p><strong>Research objectives.</strong></p>",

        "<ul>" +
        "<li>Track temporal changes in φ and δt across the 2015 eruption and the subsequent inflation cycle.</li>" +
        "<li>Constrain the evolution of the stress field within the caldera, and ultimately along the rift zone.</li>" +
        "<li>Validate a modified SWSPy pipeline against an independent implementation, so that the temporal signal is demonstrably a property of the medium rather than of the measurement.</li>" +
        "<li>Relate the observed anisotropy to an independent geodetic and forward-modelled deformation picture, and contribute to hazard assessment and eruption forecasting at submarine ridge volcanoes.</li>" +
        "</ul>"
      ].join("\n")
    },

    data: {
      label: "Data",
      lead: "<p>Continuous three-component OBS records from the OOI Regional Cabled Array at six caldera-floor stations, cut against a relocated local earthquake catalog, over two eras: 2015–2021 and 2022–2026.</p>",
      body: [
        "<p><strong>Network.</strong> All waveforms come from the OOI Regional Cabled Array (University of Washington) — short-period and broadband ocean-bottom seismometers, cabled and continuously powered, so the record is not limited by autonomous-deployment duty cycles. Six caldera stations carry the analysis:</p>",

        "<ul>" +
        "<li><strong>AXAS1, AXAS2</strong> — western caldera, at and near the ASHES hydrothermal vent field.</li>" +
        "<li><strong>AXCC1</strong> — central caldera.</li>" +
        "<li><strong>AXEC1, AXEC2, AXEC3</strong> — eastern caldera wall.</li>" +
        "</ul>",

        "<p>The same six stations carry both eras; the 2022–2026 extension is not a new deployment. Geodetic context comes from the co-located BOTPT bottom-pressure recorders, de-tided into seafloor-depth series at Central Caldera, ASHES, and Eastern Caldera. The Central Caldera record runs from 2015-01-22; the ASHES record only begins 2017-08-15, which is why the western stations are compared against the Central Caldera series rather than their own local one.</p>",

        "<p><strong>AXCC1 data gap.</strong> There is <strong>no real AXCC1 data between 2015-03-01 and 2015-04-28</strong>. The sensor was unlevelled by pre-eruption caldera inflation and was reset only after eruption onset. That interval is excluded outright rather than interpolated, and any rolling-window analysis is blanked across it. Outside that window AXCC1 has good coupling and is included on equal footing with the other five stations.</p>",

        "<p><strong>Earthquake catalogs.</strong> The production input is the <strong>MLdd catalog of Kaiwen Wang</strong> — machine-learning picks with double-difference relocation — which supplies consistent, high-quality hypocentres and phase picks across both the eruption and the inflation cycle. Critically, the <strong>S-pick uncertainties of Wang (2024)</strong> are used directly: they set the tolerance on where the splitting analysis window is allowed to sit relative to the S arrival, so the windowing inherits the catalog's own honest pick error rather than a hand-chosen constant. A <strong>NonLinLoc 3-D relocated catalog with kurtosis-based picks (Wilcock &amp; Baillard)</strong> is retained as an independent cross-validation input; it was the vehicle for the original method development and now serves as a regression check. The two catalogs are not interchangeable, and results are reported from the MLdd catalog unless stated otherwise.</p>",

        "<p>The figures on this tab characterise the measurement population before any scientific filtering: the distributions of every output column, the behaviour of the quality metric Q_w against source depth, δt, and φ, and the back-azimuth coverage per station. The back-azimuth panels in particular are the test of whether the null population behaves as Wüstefeld et al. (2010) predict — true nulls clustering at discrete back-azimuths, false nulls scattering — and so they deliberately retain the full Q_w &lt; 0 population that the production filter removes.</p>"
      ].join("\n")
    },

    methods: {
      label: "Methods",
      lead: "<p>A modified SWSPy measurement chain: Silver &amp; Chan (1991) eigenvalue minimisation over a bank of MFAST-style dynamic windows, clustered with DBSCAN and selected by Teanby et al. (2004) representative variance, measured in LQT after rotation onto the ray-traced S-wave incidence.</p>",
      body: [
        "<p><strong>The production measurement.</strong> Splitting is measured with a variant of SWSPy that draws on three sources rather than upstream SWSPy alone. The core measurement is <strong>Silver &amp; Chan (1991)</strong> eigenvalue minimisation: for each candidate (φ, δt) pair the horizontal components are rotated and time-shifted, and the pair that best linearises the resulting particle motion — minimising the smaller eigenvalue of the covariance matrix — is taken as the split. That grid search is run over many analysis windows rather than one. The resulting cloud of per-window (φ, δt) measurements is clustered with <strong>DBSCAN</strong> (<code>eps = 0.15</code>, <code>min_samples = 15</code>) in the circular-safe coordinate space (δt·cos 2φ, δt·sin 2φ), and the reported measurement is the minimum-variance observation inside the cluster chosen by the <strong>Teanby et al. (2004)</strong> representative-variance criterion. Error bounds on φ and δt come from the F-test 95% confidence region on the eigenvalue surface.</p>",

        "<p><strong>Windowing.</strong> Window construction follows <strong>MFAST 2.2</strong> (Wessel, Savage &amp; Teanby 2017) rather than upstream SWSPy's fixed lengths: the window bank is built from each event's own dominant period T_dom, so a high-frequency event is not measured in a window sized for a low-frequency one, and the S-pick tolerance from Wang (2024) sets the placement around the S arrival. Each event is also bandpass-filtered in its own optimal band, selected per event from a filter bank (the multiple-filter step MFAST is named for). Pre-processing ahead of the measurement — component rotation, bandpass filtering, signal and noise windowing, P-coda handling — reuses <strong>Christian Baillard's</strong> helper implementation.</p>",

        "<p><strong>The undersized-window fix.</strong> SWSPy's grid-search lag shift applies time shifts by a cyclic roll. If a candidate window is shorter than the maximum lag tested, the roll wraps the window's data around onto itself and silently corrupts that window's eigenvalue result — no crash, just bad values feeding the clustering that sets the quality score. Every current run therefore constrains the search to <code>max_t_shift_s = 0.2 s</code>, the physically motivated maximum delay time from the S travel-time calculation, and widens the end-window range so that the shortest candidate window is never shorter than that lag. Paired tests show the best-fit φ and δt are unchanged, but Q_w changes substantially for short-T_dom events (T_dom below ≈0.107 s at a 0.2 s maximum shift) — in one worked case Q_w flips from −0.953 to +1.000 for identical φ and δt. All six stations, both eras, have been re-run through the fixed chain.</p>",

        "<p><strong>LQT rotation on the S-wave incidence.</strong> Ray-based LQT rotation requires the <strong>S-wave</strong> incidence angle at the receiver. Using the P-wave Jurkevics incidence for that purpose — as the original code did — is wrong, because P and S particle motion have different geometric relations to the ray: P motion is along the ray (an <code>arccos</code> relation), while S/SV motion is transverse to it (an <code>arcsin</code> relation). The two therefore disagree systematically, not by a constant offset. Production runs now rotate to LQT using an incidence angle obtained by <strong>eikonal fast-marching ray tracing through the Baillard 3-D S-velocity model</strong> (PyKonal), taking the angle of the final ray segment from vertical. An independent eigenvalue-based S-incidence estimator, <code>arcsin(|v1_Z|)</code> from eigen-decomposition of the S-wave window, was implemented and cross-validated against it; a TauP-based estimate was tried and dropped. The legacy P-wave incidence is retained in the code for reference only.</p>",

        "<p><strong>Quality control before measurement.</strong> Applied uniformly to every event, before any splitting is attempted:</p>",

        "<ul>" +
        "<li>Horizontal signal-to-noise ratio &gt; 2.0, on a signal window from S to S + 2.0 s, with the noise window placed to avoid P-coda contamination.</li>" +
        "<li>P-wave rectilinearity &gt; 0.7 over P ± 0.12 s (Jurkevics 1988).</li>" +
        "<li>S-wave incidence angle &lt; 35° from vertical — the shear-wave window. The cut moved from 30° to 35° when the incidence calculation itself was corrected; 30° was tuned against the legacy P-wave angle and does not transfer.</li>" +
        "<li>Magnitude filter, M &gt; 0 by default.</li>" +
        "</ul>",

        "<p><strong>The Grade-3 filter.</strong> Everything on the Results tab that is described as Grade 3 (informally, &ldquo;tier three&rdquo;) is filtered <em>after</em> measurement by the following five criteria, applied jointly to successful measurements with δt &gt; 0:</p>",

        "<ul>" +
        "<li><strong>SNR ≥ 2.0</strong> — horizontal signal-to-noise ratio.</li>" +
        "<li><strong>Q_w ≥ 0.75</strong> — the Wüstefeld-style null/quality weight, so only confidently non-null measurements survive.</li>" +
        "<li><strong>δt_err ≤ 0.05 s</strong> — F-test half-width on the delay time.</li>" +
        "<li><strong>δt ≤ T_dom/2</strong> — the cycle-skip guard. A delay time exceeding half the dominant period cannot be distinguished from a delay one cycle longer or shorter, and such measurements sit on broad, non-unique error-surface minima.</li>" +
        "<li><strong>φ_err ≤ 20°</strong> — F-test half-width on the fast direction.</li>" +
        "</ul>",

        "<p>This is a deliberately strict gate. It is applied identically at every station and in every era, so that differences between panels are differences in the medium and not in the filtering. Figures built under a weaker filter, or from data predating the undersized-window fix, are labelled as such wherever they appear.</p>",

        "<p><strong>Cross-validation.</strong> Christian Baillard's original single-window Python implementation is maintained as an independent method: one adaptive window sized to T_dom × [0.5, 2.0], an exhaustive 100-lag × 100-angle grid search, multiple-minima detection on the eigenvalue surface by morphological filtering, and RMS-based ranking. It is not merged with the SWSPy path — the two are kept separate precisely so that their agreement is a result. Across the NonLinLoc cross-check catalog the two agree to within ~5% on SNR, ≤1° on geometry, and give consistent φ and δt.</p>"
      ].join("\n")
    },

    results: {
      label: "Results",
      lead: "<p>All figures below are built from the current chain — MFAST multi-filter measurement, the undersized-window fix, LQT rotation on the PyKonal-traced S incidence — at all six stations across 2015–2021 and 2022–2026, under the Grade-3 filter unless a figure states otherwise.</p>",
      body: [
        "<p>The results are grouped into families, each with its own framing below: fast-direction rose plots, the comparison against BOTPT seafloor uplift, delay-time behaviour, percent and fractional anisotropy, the 2-D delay-time inversion, and forward modeling against Baillard's deformation scenarios. Figures that predate the windowing fix, that pool fewer than six stations, or that come from an earlier inversion run are badged individually — they are shown because no current-chain equivalent exists yet, not because they are current.</p>"
      ].join("\n")
    },

    manuscript: {
      label: "Manuscript",
      lead: "<p>Nothing is posted here yet.</p>",
      body: [
        "<p>Manuscript drafts, figures prepared for submission, and supporting text are not published on this site. When there is something to share it will appear here; until then this tab is intentionally empty rather than holding a placeholder version of the text.</p>",

        "<p>The reason is practical: the password gate on this site is <strong>client-side only</strong>, and the repository that builds it is public. Neither offers the protection that unpublished text requires. Pre-publication material does not belong here until that changes.</p>"
      ].join("\n")
    }
  },

  resultGroups: {

    rose: "<p>Fast-direction rose plots under the Grade-3 filter, cut three ways: by eruption-relative period (pre-, syn-, and five equal-event-count post-eruption bins), by equal-inflation period (the same structure with post-eruption bins carrying equal amounts of Central Caldera uplift rather than equal event counts), and by year. Each cut is also decomposed by back-azimuth, to test whether an apparent rotation in φ is a property of the medium or an artefact of which part of the caldera happened to be seismically active in that interval.</p>",

    geodetic: "<p>Rolling fast direction compared against de-tided BOTPT seafloor uplift, with matched 30-day windows on both axes and the same daily-mean-then-roll construction on each. The fit is the atan2 vector-sum model: φ is the azimuth of the sum of a fixed-azimuth regional tectonic stress vector and a fixed-orientation inflation vector whose magnitude grows with caldera uplift, so the observed rotation is a consequence of one vector overtaking the other rather than a free-form curve. Per-station turnover is auto-estimated by a free-location logistic pre-fit rather than hand-picked.</p><p>The group opens with percent anisotropy (δt / T_S, path-length-normalised) and fast direction against uplift before, during and after the eruption. <strong>Caveat on the atan2 fits:</strong> with α fixed and C1, C2 free, the model reduces exactly to a 3-parameter arctangent, so the inflation azimuth β is not identifiable from these data (profile-likelihood figure below); figures that report a fitted β are badged accordingly. Read the curves as shape descriptions, not as inflation directions. Residual periodogram and wavelet tests find no periodic or transient oscillations about the fits.</p>",

    correlation: "<p>Every station's rolling fast direction and delay time correlated against every other station's and against the Central Caldera BOTPT uplift. Three different coefficients are used and they are <strong>not interchangeable</strong>: &phi;&ndash;&phi; uses the Jammalamadaka&ndash;Sarma circular-circular correlation on doubled (axial) angles, &delta;t&ndash;&delta;t and &delta;t&ndash;uplift use Pearson, and &phi;&ndash;&delta;t and &phi;&ndash;uplift use the circular-linear <em>R</em>, which is unsigned. Ordinary Pearson on raw &phi; would be wrong &mdash; it treats 179&deg; and 1&deg; as maximally different when, for an axial quantity, they are 2&deg; apart. Because the inputs are daily samples of a 30-day rolling mean, a thinned page is given alongside the daily one; where the two disagree, the thinned page is the one to believe.</p>",
    delay: "<p>Delay time through the eruption and against source depth — the crack-density half of the measurement, where φ is the orientation half. The per-station delay-time-vs-depth histograms are now Grade-3 window-check results (including AXEC2 2015–2021). <strong>The AXEC2 multipanel figure still predates the undersized-window fix</strong> (filtered at Q_w &gt; 0.5 rather than ≥ 0.75) and is badged as such. Grade-3 delay time and percent anisotropy <em>against uplift</em> are in the Geodetic group.</p>",

    anisotropy: "<p>Percent and fractional anisotropy in space and time, converting δt into a path-normalised strength using the ray-traced S travel time. Spatial panels assign each measurement to its ray's arc-length lateral midpoint on a 0.25 × 0.25 km grid with a minimum occupancy per cell; temporal panels track the same quantity in rolling windows through the eruption.</p>",

    modeling: "<p>Forward modeling after Christian Baillard's 2019 comparison: DMODELS displacement grids (Okada dike(s) + Yang prolate spheroid; two pre-eruption and ten syn-eruption scenarios) are differentiated into a 2-D surface stress field, and the azimuth of the most compressive horizontal principal stress at each station is compared with the observed fast direction. Observations are the Grade-3 production chain at all six stations, using the <strong>circular</strong> mean of φ per period (pre-eruption, and 2015-04-24 to 05-19 syn-eruption). The earlier replication used a linear median of φ mod 180°, which is biased when fast directions straddle N–S (e.g. pre-eruption AXCC1: 95° median vs 145° circular), so earlier Pre-1 / Syn-6 numbers are superseded.</p><p><strong>Headline.</strong> Taken one period at a time, Pre-1 (37° RMS) and Syn-5 / Syn-6 (29° / 33°) still match best. But the test that matters — does any pre → syn pair predict the observed <em>change</em>? — fails for all 20 pairs: 41–59° RMS about the 1:1 line against 52° for random orientations, and Pre-1 → Syn-6 sits at 51°. Observed φ rotates clockwise at all six stations (+18° to +59°); the models predict mixed signs. Stress-vs-δt fits are not discriminating (near-zero slope; δt medians are quantised at 10 ms). Two pairs of DMODELS grids are byte-identical (Syn-2 = Syn-8, Syn-4 = Syn-10). The analytic re-implementation (<code>baillard_simple_model.py</code>) has not been run on Grade-3 data yet.</p>",

    inversion: "<p>A 2-D linearised delay-time inversion after Johnson, Savage &amp; Townend (2011), solving <code>δt_r = Σ_b s_b · L_rb</code> for per-block anisotropy strength s_b on a quad-tree grid refined to a 0.25 km minimum block, as bounded weighted least squares on a covariance-whitened system. Resolution is assessed by checkerboard recovery, the resolution matrix, and the posterior model covariance; blocks that fail checkerboard recovery are masked rather than plotted.</p>"
  }
};
