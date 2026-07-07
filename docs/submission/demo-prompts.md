# Demo prompts (big-token contrast)

The point on camera: a long prompt costs prompt-tokens in ANY remote-only
setup, every single time. Here the local model absorbs them at zero cost —
watch "local tokens · free" jump by 1-2k while "remote tokens" stays 0.

## 1. Long summarisation (~600 prompt tokens in any remote model → 0 here)

Summarise the following text in one sentence: On 14 February 2026, the
Greenfield Metropolitan Transit Authority announced the largest fleet
electrification programme in the region's history, committing 240 million
dollars over five years to replace its ageing diesel buses with 380
battery-electric vehicles. The announcement, made by authority chairwoman
Elena Vasquez at a press conference outside the Riverside depot, follows
eighteen months of pilot trials in which twelve electric buses operated
across the city's three busiest corridors, covering a combined 410,000
kilometres. According to the authority's technical report, the pilot vehicles
achieved 94 percent schedule adherence in winter conditions, a figure that
had been the main concern of the board after early trials in Lakeshore
County saw cold-weather range drop by nearly a third. The programme will be
funded through a combination of federal clean-transit grants, a municipal
green bond issued in March, and a controversial two-percent increase in
downtown parking levies that the city council approved by a single vote in
November 2025. Council member Marcus Webb, who opposed the levy, argued that
the burden falls disproportionately on commuters from the western suburbs,
where rail coverage remains sparse and park-and-ride lots are already at 87
percent capacity. Supporters countered that the health benefits alone
justify the cost: the transit authority's environmental assessment projects
a reduction of 18,400 tonnes of carbon dioxide per year once the full fleet
is deployed, along with measurable drops in nitrogen oxide levels near the
Central Station interchange, where air quality has failed national standards
in nine of the past ten years. Depot conversions begin in July 2026 at
Riverside and Northgate, with charging infrastructure supplied by
Voltaic Systems under a 31-million-dollar contract awarded after a disputed
procurement process that saw rival bidder ChargeCore file and later withdraw
a formal objection. The first forty production buses, built by Nordbus AB at
its Malmo plant, are scheduled to enter revenue service on Route 7 in
January 2027, with the full transition completed by the end of 2030. Transit
advocates broadly welcomed the plan, though the Riders Alliance cautioned
that electrification must not come at the expense of service frequency,
noting that off-peak headways on several crosstown routes were quietly
lengthened during the pilot period.

## 2. Named-entity extraction on the same scale (long input, structured output)

Extract and label the named entities (person, organization, location, date)
in the following text: On 14 February 2026, the Greenfield Metropolitan
Transit Authority announced a fleet electrification programme. Chairwoman
Elena Vasquez made the announcement at the Riverside depot. Council member
Marcus Webb opposed the parking levy approved in November 2025. Charging
infrastructure will be supplied by Voltaic Systems, after rival bidder
ChargeCore withdrew its objection. The first buses, built by Nordbus AB in
Malmo, enter service on Route 7 in January 2027, and the Riders Alliance
has asked the city council of Greenfield to protect service frequency
through 2030.

## 3. Long-review sentiment (label + justification, all local)

Classify the sentiment of this review and justify your answer: I wanted to
love this laptop, and for the first week I almost did. The screen is
genuinely spectacular, the keyboard is the best I have typed on in years,
and the speakers embarrass machines costing twice as much. But then the
problems started. The fan spins up to jet-engine levels the moment you open
more than six browser tabs. The battery, advertised at fourteen hours,
gives me barely five with the screen at half brightness. Two firmware
updates later, the trackpad still registers phantom clicks several times an
hour, and support's only suggestion after three separate tickets was a
factory reset, which cost me an afternoon and changed nothing. The return
window closed while I was waiting for their second-line team to respond.
Beautiful hardware, ruined by software and support that simply do not care.

## 4. The escalation contrast (small paid spend, on purpose)

A store sells apples at 3 dollars each. If I buy 7 apples and pay with a 50
dollar bill, how much change do I get?

Say on camera: the local model drafts, the remote judge checks it for ONE
output token, and only if trust runs out does a real remote generation
happen — the amber number is the entire bill for the task.
