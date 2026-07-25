Customer: Hi, I'm trying to export our transaction history for the last quarter
but the download keeps failing at around 80%. I've tried Chrome and Safari.

Agent: Thanks for reporting that. Could you tell me roughly how many
transactions are in the range you're selecting? There's a known issue with
synchronous exports above about fifty thousand rows.

Customer: Let me check. It looks like around 90,000 for the quarter.

Agent: That would explain it. Exports above fifty thousand rows need to go
through the asynchronous path, which emails you a link when the file is ready.
The synchronous download times out at the gateway before the file finishes
generating. On the export screen there's a checkbox labelled "email when ready"
which switches to that path.

Customer: I see it now. Does that have a row limit too?

Agent: It's capped at two million rows per export. Above that you'd want to
split by month, or use the bulk export API which streams and has no cap.

Customer: The API sounds better long term since we want to automate this. Where
do I find the docs?

Agent: The bulk export endpoints are under the Reporting section of the API
reference. You'll need a key with the reporting:read scope, which an
organisation admin can grant from the API keys page. The endpoint paginates with
a cursor, and the cursor stays valid for twenty-four hours.

Customer: Perfect, I'll set that up. Thanks for the quick help.

Agent: Happy to help. I'll also add a note to your account about the export
size so anyone picking up a future ticket has the context.
