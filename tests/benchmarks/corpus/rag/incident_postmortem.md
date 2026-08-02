On 14 July 2026 the payments service experienced a partial outage lasting
thirty-eight minutes, during which approximately 4% of card authorisation
requests failed with a gateway timeout. The incident began at 09:12 UTC when a
routine configuration change was applied to the connection pool settings on the
primary authorisation cluster.

The change reduced the maximum pool size from 240 connections to 60. This value
had been copied from a staging manifest during a templating refactor, and the
reviewer did not catch the discrepancy because the diff presented the value as
part of a larger reformatting change. Automated tests did not detect the problem
because the staging environment never sustains enough concurrent load to exhaust
a 60-connection pool.

Traffic at 09:12 was near the daily peak. Within ninety seconds the pool was
saturated and requests began queuing. The queue depth metric was being collected
but was not part of any alerting rule, so the first signal reached the on-call
engineer only when the customer-facing error rate crossed its threshold at
09:19, seven minutes after the change landed.

Diagnosis was slower than it should have been. The on-call engineer initially
suspected the upstream card network because the error signature was a timeout
rather than a connection refusal. Two engineers spent eleven minutes examining
network-level telemetry before a third noticed the deployment marker on the
dashboard and correlated it with the onset of errors.

Rollback was straightforward once the cause was identified. The previous
configuration was reapplied at 09:44 and error rates returned to baseline within
four minutes as queued requests drained. No data was lost. Approximately 11,400
authorisation attempts failed and were retried successfully by client
integrations; an estimated 320 were abandoned by end users.

Several factors made this incident worse than it needed to be. First, the
connection pool size was expressed as an absolute number rather than derived
from expected concurrency, so an incorrect value looked plausible in review.
Second, queue depth was collected but never alerted on, which meant the system
knew it was in trouble four minutes before anyone was told. Third, deployment
markers were present on dashboards but not surfaced in the alert payload, so the
correlation that eventually solved the incident depended on someone happening to
look at the right panel.

Remediation items agreed in the review: derive pool sizing from a concurrency
target rather than hardcoding it; add an alert on sustained queue depth above
twenty; include recent deployment identifiers in every alert notification; and
extend the load profile in pre-production so that pool exhaustion is reachable
in automated testing. The first three were completed within a fortnight. The
fourth is tracked as a larger piece of work against the platform roadmap for the
following quarter.
