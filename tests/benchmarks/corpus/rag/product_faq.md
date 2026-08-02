Our subscription tiers are billed monthly in advance. The Starter plan includes
up to five seats and 10,000 API calls per month. The Growth plan raises this to
twenty-five seats and 250,000 calls, and adds priority support with a four-hour
first-response target during business hours. The Enterprise plan is negotiated
individually and includes a dedicated success contact, custom data residency,
and a contractual uptime commitment.

Changing plans mid-cycle is prorated. If you upgrade, the unused portion of your
current term is credited against the new plan and the difference is charged
immediately. If you downgrade, the credit is applied to future invoices rather
than refunded, because our payment processor charges a fee on refunds that would
otherwise be passed on.

API calls are counted per request, not per record returned. A request that
returns a page of one hundred results counts as one call. Requests that fail
with a 4xx status are counted; requests that fail with a 5xx status are not,
since those represent our error rather than yours. Rate limits are applied per
organisation rather than per key, so issuing additional keys does not increase
your quota.

Data retention defaults to ninety days on Starter and Growth, and is
configurable between thirty days and seven years on Enterprise. Deleted records
are removed from primary storage immediately and purged from backups within
thirty-five days. We do not use customer data to train models, and this is
stated contractually rather than only as policy.

Single sign-on via SAML is available on Growth and Enterprise. SCIM provisioning
is Enterprise only. Both are configured from the organisation settings page and
require a verified domain. Verification is done by DNS TXT record and usually
completes within an hour.
