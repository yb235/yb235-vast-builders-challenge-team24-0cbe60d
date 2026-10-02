---
name: submission
description: >-
  Walk a team through their VAST Builders Challenge submission section by section, propose
  or ask for each value, get their agreement, then write the final SUBMISSION.md so it's
  ready to copy and paste. Never collects personal details. Use when a team is ready to
  submit, or wants to check what is still missing.
---

# Submission

Walks the team through each section, proposing what can be derived from their code and
asking for what can't, and writes `SUBMISSION.md` only once each value is agreed. The
result is a finished file the team can copy values from, not a blank form and not
something already submitted.

Run it early to see what's outstanding, and again at the end to produce the final file.

## Before you submit

- [ ] Your project runs, from a clean start, without you fixing it live
- [ ] Your code is pushed somewhere the judges can open
- [ ] One person can explain the project in two minutes

## Work it out first

Derive the **project details** only: what the project does, and the technology stack.

Read the team's code. What it does should come from what the code actually does, not from
how they describe it in a hurry at the end of the day. For the stack, note which skills they
called, which parts of the pipeline that implies (Cosmos Reason for captions, Cosmos Embed
for search, YOLO for detection), whether they called a W&B model, and what they built the
app itself with.

Keep the description to two or three sentences, under 40 words: what it does, and who
would use it. The stack is a short list, not prose. Judges read a lot of these.

Draft both, show them in as few words as you need, and let them correct it. A team that
has been building for eight hours writes a better description by editing yours than by
starting from a blank prompt. Don't explain your reasoning or add caveats; just show the
draft and let them edit it.

The team number comes from `$USERNAME` (or `$PIPELINE`) in the environment; never ask
for it. Ask for everything else: links and feedback. Those are not in the code, and a
wrong value that looks right gets submitted without anyone noticing.

If you aren't sure, leave it empty and ask.

## What to collect

Go section by section. Present what you already worked out, ask for the gaps, and wait for
an answer before moving to the next section. Don't dump all sections at once.

**Team**
- Team number: read from `$USERNAME` or `$PIPELINE`, never asked

Never ask for, collect, or write personal details: member names, emails, ages, contact or
consent confirmations. There is no field for these, for now; don't add one even if a team
offers the information.

**Project**
- Description: what it does and who it's for, in two or three sentences
- Technology stack: which skills, which models, what you built on top
- Link to the code or repo
- Link to the live app, if it's deployed somewhere reachable (optional; not every team will
  have one running past the event)

<!-- TODO: where the code goes (GitHub or any public URL) isn't decided yet. -->
- Links to supplementary material, if any

**Feedback**
- Overall feedback on the day, a sentence or two. Invite them to call out anything specific,
  but don't ask about each product in turn. Honest criticism is more useful than praise.

## Nothing should be left blank

This is a contest entry. An incomplete one may not be judged, and the team won't find out
until it's too late to fix.

Never invent a value, but never quietly accept a blank either. If a field is missing, say
what it is and why it matters, then ask again. Two of them are worth pushing on:

- **The code link.** Judges assess what they can open. A project nobody can run is
  judged on its description alone.

When you finish, list what's still missing and tell them the entry isn't complete. Don't
end on a summary that reads like success when fields are still empty.

## Writing the file

Write `SUBMISSION.md` in the repo root using this shape, once every section is agreed.
Leave a field as `NOT PROVIDED` rather than inventing a value.

```markdown
# <Team number>

## Project
<description>

**Stack:** <stack>
**Code:** <url>
**Live app:** <url, or none>
**Supplementary:** <urls, or none>

## Feedback
<overall feedback, with anything product-specific they mentioned>

```

There is no Team/Members section. Personal details never appear in this file.

## Agent instructions

1. Never ask for, collect, or write personal details: member names, emails, ages, contact
   or consent confirmations. There is no field for these.
2. Read `SUBMISSION.md` first if it exists, and only ask for what's missing.
3. Derive only the project description and technology stack, and show them for correction.
   Ask for everything else.
4. Take one section at a time: team number, then project, then feedback. Get agreement on a
   section's values before moving to the next; only write the file once all are agreed.
   Keep every message short: show the draft or ask the question, nothing else. Let the team
   edit what you show rather than writing at length yourself.
5. Never invent a value. When unsure, leave the field empty rather than guessing.
6. Write `SUBMISSION.md` but do not commit or push it unless asked.
7. When you finish, list anything still `NOT PROVIDED`, say the entry is incomplete, and
   offer to fill the gaps now. Confirm the checklist at the top of this skill.

## After the file is written

Tell the team the file holds the values to copy into wherever the actual submission goes.
Where that is isn't decided yet, so don't tell them the entry is submitted, only that it's
ready to copy from.

<!-- TODO: no submission destination exists yet. Confirm where a team actually submits
     (paste into a form, upload the file, post a link) and where. Also confirm with
     whoever owns the Terms & Conditions whether confirmations collected this way are
     sufficient, or whether the official form must still be signed separately. -->
