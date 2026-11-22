"""Phrase banks for the synthetic scouting notes.

This file is data, kept apart from the generator logic so it can be read
on its own. Each bank maps a true code to sentences that express it.

  plain    sentences that state the code directly
  hard     sentences that need a codebook rule to resolve (sitting velocity
           versus peak, raw versus game power, a negated concern, a decoy
           role statement, and so on). Notes flagged ambiguous use at
           least one hard sentence.
  hedged   health mentions that are unconfirmed but still code "yes"

Every list has a "_new" partner (plain_new, hard_new, hedged_new, ...).
Sentences in a _new list are held out of dev notes: they appear only in
test notes. A keyword list written from dev notes has never seen them,
as it would not have seen a new season's wording. The generator records,
for every note, which fields used held-out wording (column heldout_fields).

Placeholders in braces are filled by generate_notes.py:
  {hand} RHP or LHP     {ht} height like 6-2     {wt} weight in pounds
  {pos} position code   {noun} position name     {side} right, left or both
  {lo} {hi} {peak} fastball velocities in mph    {asset} arm or bat
No real players, scouts, teams or schools appear anywhere.
"""

POSITIONS = {
    "C": "catcher",
    "SS": "shortstop",
    "2B": "second baseman",
    "3B": "third baseman",
    "CF": "center fielder",
    "LF": "left fielder",
    "RF": "right fielder",
    "1B": "first baseman",
}

# Opening sentence: always carries the role.
OPENERS = {
    "starter": {
        "plain": [
            "{hand}, {ht}, {wt}, has started every weekend this spring.",
            "{hand} who has held up as a starter, working six innings most outings.",
            "{hand}, {ht}, {wt}, profiles as a starter with a three-pitch mix.",
        ],
        "plain_new": [
            "{hand} starter, {ht}, {wt}, turns a lineup over two or three times.",
            "{hand}, {ht}, {wt}, has gone deep into games in the weekend rotation.",
        ],
        "hard": [
            "{hand}, {ht}, {wt}, has pitched out of the pen this year, but we see "
            "a starter once the third pitch comes along.",
            "{hand}, {ht}, {wt}, has opened some games and finished others, and "
            "the plan long term is to keep him in the rotation.",
        ],
        "hard_new": [
            "{hand}, {ht}, {wt}, closes for his summer team, but the plan is to "
            "stretch him out as a starter next year.",
        ],
    },
    "reliever": {
        "plain": [
            "{hand}, {ht}, {wt}, works in short relief out of the pen.",
            "{hand} who comes out of the bullpen for an inning at a time.",
            "{hand}, {ht}, {wt}, fits best in relief with a two-pitch mix.",
        ],
        "plain_new": [
            "{hand} reliever, {ht}, {wt}, has closed games for his summer team.",
            "{hand}, {ht}, {wt}, gets the ball in the ninth for his summer team.",
        ],
        "hard": [
            "{hand}, {ht}, {wt}, starts for his team now, but the effort in the "
            "delivery and the two-pitch mix point to the bullpen.",
            "{hand}, {ht}, {wt}, has made a few starts, though we see him as a "
            "late-inning arm long term.",
        ],
        "hard_new": [
            "{hand}, {ht}, {wt}, is in the weekend rotation, though his stuff "
            "plays best in short bursts and we see him in relief.",
        ],
    },
    "position_player": {
        "plain": [
            "{pos}, {ht}, {wt}, bats {side}.",
            "{pos}, {ht}, {wt}, everyday {noun} for his summer team.",
        ],
        "plain_new": [
            "{Noun} who has played every day this spring, {ht}, {wt}.",
            "{Noun}, {ht}, {wt}, hits near the top of the order.",
        ],
        "hard": [
            "Two-way player, {ht}, {wt}, who also pitches, but we like him better "
            "as an everyday {noun}.",
        ],
        "hard_new": [
            "Has also thrown a few innings on the mound, {ht}, {wt}, but his "
            "future is as an everyday {noun}.",
        ],
    },
}

# Fastball velocity for pitchers. The {lo} to {hi} range is drawn so that it
# matches the code under the codebook thresholds.
VELO = {
    "below_average": {
        "plain": [
            "Fastball sits {lo} to {hi}.",
            "Fastball is fringy and plays light.",
            "Below-average velocity, mostly {lo} to {hi}.",
            "Fastball is a 45 now.",
        ],
        "plain_new": ["Velocity is short, sits {lo} to {hi}.", "Below-average fastball without much life."],
        "hard": ["Lit up the gun at {peak} once, but sat {lo} to {hi} most of the night."],
        "hard_new": ["Hit {peak} on one radar reading, but the fastball lived at {lo} to {hi} otherwise."],
    },
    "average": {
        "plain": [
            "Fastball sits {lo} to {hi}.",
            "Fastball is average, nothing loud.",
            "Solid-average velocity at {lo} to {hi}.",
            "Fastball grades as a 50.",
        ],
        "plain_new": [
            "Fastball sits {lo} to {hi} with some sink.",
            "Average arm strength, the fastball lives at {lo} to {hi}.",
        ],
        "hard": ["Touched {peak} in the first inning but sat {lo} to {hi} after that."],
        "hard_new": ["Opened the game at {peak} and settled in at {lo} to {hi}."],
    },
    "plus": {
        "plain": [
            "Fastball sits {lo} to {hi}, touching {peak}.",
            "Plus fastball with life up in the zone.",
            "Above-average heater that sits in the mid 90s.",
            "Fastball is a plus-plus pitch at its best.",
            "Fastball is a 60 pitch.",
        ],
        "plain_new": [
            "Heater has late life and sits {lo} to {hi}.",
            "Fastball flashes plus-plus and sits {lo} to {hi}.",
        ],
        "hard": ["Velocity dipped late, but he sat {lo} to {hi} for five innings."],
        "hard_new": ["Gun readings fell off late in the outing, but he held {lo} to {hi} through the sixth."],
    },
}

# Raw power for position players.
POWER = {
    "below_average": {
        "plain": [
            "Below-average raw power, more of a slap hitter.",
            "Fringy pop, mostly singles.",
            "Power is light, a 40 now.",
        ],
        "plain_new": ["Not much impact in the bat, below-average raw.", "Bat is light, more contact than damage."],
        "hard": ["In-game power has been fine so far, but the raw power is below average."],
        "hard_new": ["Can hit a mistake out to the pull side, but the raw power is fringy."],
    },
    "average": {
        "plain": [
            "Average raw power, more gap to gap.",
            "Solid-average pop to the pull side.",
            "Raw power grades as a 50.",
        ],
        "plain_new": ["Average raw, can run into one.", "Power is average, more doubles than homers."],
        "hard": ["Hit a couple out this spring, but the raw power is only average."],
        "hard_new": ["Raw power is average even though the game power has been loud this spring."],
    },
    "plus": {
        "plain": [
            "Plus raw power, the ball jumps off the bat.",
            "Plus-plus raw, carries to all fields.",
            "Above-average power that shows up in games.",
            "Raw power is a 60.",
        ],
        "plain_new": ["Flashes plus raw power to the pull side.", "Easy plus raw with carry to the opposite field."],
        "hard": ["Plus raw power in batting practice, though it plays fringy in games so far."],
        "hard_new": ["Game power lags behind, but the raw power is plus when he gets extended."],
    },
}

COMMAND = {
    "concern": {
        "plain": [
            "Walks too many and loses the zone in long innings.",
            "Spotty command, falls behind hitters.",
            "Control is below average right now.",
        ],
        "plain_new": ["Fringy control, too many deep counts.", "Misses arm side often and runs up pitch counts."],
        "hard": ["Throws strikes, but they are not quality strikes and catch too much of the plate."],
        "hard_new": ["Fills the zone, but leaves too many pitches over the middle."],
    },
    "neutral": {
        "plain": [
            "Throws enough strikes.",
            "Average control, nothing alarming.",
            "Command is fine for his age.",
        ],
        "plain_new": ["Control is average.", "Strike-throwing is about average."],
        "hard": ["No concerns with his strike-throwing."],
        "hard_new": ["Nothing to worry about with his control."],
    },
    "strength": {
        "plain": [
            "Plus command, paints both sides of the plate.",
            "Advanced feel for locating all three pitches.",
            "Pounds the zone and rarely walks anyone.",
        ],
        "plain_new": ["Command is ahead of his age, works both corners.", "Locates to both sides of the plate with ease."],
        "hard": ["Never seems to fall behind, works ahead of almost every hitter."],
        "hard_new": ["Hardly ever gets behind in the count and spots the fastball where he wants it."],
    },
}

APPROACH = {
    "concern": {
        "plain": [
            "Free swinger who chases breaking balls away.",
            "Expands the zone with two strikes.",
            "Approach is raw, lots of swing and miss in the zone.",
        ],
        "plain_new": ["Chases out of the zone too often.", "Swings at the first pitch he sees, too aggressive."],
        "hard": ["Draws some walks, but mostly because pitchers avoid him, and he chases when they attack."],
        "hard_new": ["Walk totals look fine against weak pitching, but he expands against good arms."],
    },
    "neutral": {
        "plain": [
            "Approach is fine, average pitch recognition.",
            "Swing decisions are average.",
            "Handles the zone well enough.",
        ],
        "plain_new": ["Average feel for the zone.", "Pitch selection is about average."],
        "hard": ["No real concerns with his approach."],
        "hard_new": ["Nothing alarming about his approach."],
    },
    "strength": {
        "plain": [
            "Patient hitter who works counts and rarely chases.",
            "Advanced approach, controls the zone.",
            "Plus pitch recognition, takes his walks.",
        ],
        "plain_new": ["Mature approach, rarely gives away an at-bat.", "Works deep counts and lays off breaking balls."],
        "hard": ["Rarely expands the zone, even with two strikes."],
        "hard_new": ["Seldom offers at pitches off the plate."],
    },
}

MAKEUP = {
    "positive": {
        "plain": [
            "Coaches rave about his work ethic.",
            "High-character kid who competes.",
            "Plus makeup, first one at the field.",
            "Teammates follow him, a real leader.",
        ],
        "plain_new": ["Grinder, plays hard every inning.", "Coaches call him the hardest worker on the team."],
        "hard": ["Never heard a bad word about him from his coaches."],
        "hard_new": ["Not one coach had anything negative to say about him."],
    },
    "negative": {
        "plain": [
            "Makeup questions, body language sags when things go wrong.",
            "Heard concerns about his work ethic from more than one coach.",
            "Can be hard to coach, does not take adjustments well.",
        ],
        "plain_new": ["Effort comes and goes, and his coaches have noticed.", "Has clashed with coaches over effort."],
        "hard": ["Says the right things, but his coaches say he does not put in the work."],
        "hard_new": ["Talks a good game, but coaches say his work habits are poor."],
    },
    # not_mentioned has no plain sentence: the note simply says nothing.
    "not_mentioned": {
        "plain": [], "plain_new": [],
        "hard": ["Quiet kid who keeps to himself."],
        "hard_new": ["Soft-spoken, lets others do the talking."],
    },
}

HEALTH = {
    "yes": {
        "plain": [
            "Missed three weeks with a sore elbow.",
            "Had elbow ligament surgery as a sophomore.",
            "Shut down for a month with shoulder soreness.",
            "Missed most of April with a hamstring strain.",
        ],
        "plain_new": ["Coming back from a broken wrist.", "Missed the summer after knee surgery."],
        # Hedged mentions still code yes. The simulated coders tend to miss these.
        "hedged": [
            "May have had some forearm tightness last summer, not confirmed.",
            "Heard he was dealing with a sore shoulder, could not confirm it.",
        ],
        "hedged_new": [
            "There was talk of a back issue in the fall, worth checking.",
            "Might have played through some hip soreness, nobody would say for sure.",
        ],
        "hard": ["Left a game early with what the team called cramping."],
        "hard_new": ["Sat out a weekend series with what his coach called a tired arm."],
    },
    "no": {
        # Said only to rule out a problem: still codes no.
        "plain": ["Medicals are clean.", "No injury history that we know of."],
        "plain_new": ["Healthy all spring, no red flags.", "Has not missed a game with an injury."],
        # Hard "no" sentences mention the body but rule out a problem.
        "hard_pitcher": ["Some effort in the delivery, but nothing that worries us physically."],
        "hard_pitcher_new": ["Delivery has some violence, but no arm issues have come up."],
        "hard_hitter": ["Wore a knee brace in one game, which his coach says is only a precaution."],
        "hard_hitter_new": ["Had his hand looked at after a foul ball, and it turned out to be nothing."],
    },
}

# Projection comes as a full sentence or as a clause added to the opener.
PROJECTION = {
    "high": {
        "plain": [
            "Very projectable frame with lots of room to fill.",
            "Long, lean body that should add plenty of strength.",
            "Plenty of projection left in the frame.",
        ],
        "plain_new": ["Projects to add strength, lots of room on the frame.", "Frame has a lot of room to grow."],
        "clause": ["with a very projectable frame"],
        "clause_new": ["with plenty of projection left", "with a frame that has lots of room to grow"],
        "hard": ["Young for the class with a narrow frame, and the strength should come."],
        "hard_new": ["Still one of the youngest in the class, with a thin frame that should fill out."],
    },
    "some": {
        "plain": [
            "Some projection left in the frame.",
            "Can add a bit more strength as he matures.",
            "Moderate room to fill out.",
        ],
        "plain_new": ["Should add a little more strength.", "A bit more growth left in the body."],
        "clause": ["with some projection left"],
        "clause_new": ["with room to add a bit more strength"],
        "hard": ["A little room left on the frame, though he is already thick through the lower half."],
        "hard_new": ["Has a bit of room left to add strength, though he is already solid."],
    },
    "limited": {
        "plain": [
            "Body is close to maxed out.",
            "Already filled out, what you see is what you get.",
            "Physically mature, not much projection left.",
        ],
        "plain_new": ["Maxed-out frame, no projection to speak of.", "Fully grown, no more physical upside."],
        "clause": ["with a body that is close to maxed out"],
        "clause_new": ["with a fully grown frame"],
        "hard": ["Already 23, so the body is what it is."],
        "hard_new": ["At 22 he is older for the class, and the frame is set."],
    },
    # A decoy: a future-role statement, which the codebook says is not projection.
    "not_stated": {
        "plain": [], "plain_new": [], "clause": [], "clause_new": [],
        "hard_by_role": {
            "starter": "Projects as a back-end starter if the changeup comes.",
            "reliever": "Projects as a middle reliever.",
            "position_player": "Projects as a utility type.",
        },
        "hard_by_role_new": {
            "starter": "Projects as a mid-rotation arm if the slider tightens.",
            "reliever": "Projects as a setup man.",
            "position_player": "Projects as an everyday regular at a corner.",
        },
    },
}

CLOSING = {
    "positive": {
        "plain": [
            "Follow closely.",
            "Worth pushing for next spring.",
            "Would like to see him again soon, a priority follow.",
            "Strong follow.",
        ],
        "plain_new": ["Push for him.", "Keep him high on the follow list."],
        "hard": ["Hard to walk away from this {asset}."],
        "hard_new": ["Would be hard to pass on this {asset}."],
    },
    "negative": {
        "plain": [
            "Pass.",
            "Not a follow for us.",
            "Hard to see a path, would not spend more time here.",
            "No follow.",
        ],
        "plain_new": ["Organization player at best, no follow.", "Nothing here for us."],
        "hard": ["We would not spend another trip on him."],
        "hard_new": ["Not someone we need to see again."],
    },
    "mixed": {
        "plain": [
            "Intriguing but raw, need another look before deciding.",
            "Some tools, some holes, a wait-and-see follow.",
            "Good and bad here, see him again before making a call.",
            "Tough read, the decision can wait until summer.",
        ],
        "plain_new": [
            "Flashes enough to keep an eye on, but the holes are real.",
            "Some promise, some doubt, worth one more look.",
        ],
        "hard": ["Would not be shocked if he figures it out, but would not push for him either."],
        "hard_new": ["Would not rule him out, but would not chase him either."],
    },
}

# Context sentences carry no codable content. They make notes less uniform
# and are used in dev and test alike.
CONTEXT = [
    "Saw him twice this spring.",
    "Seen at a summer showcase.",
    "Saw him in a doubleheader in May.",
    "Two looks this spring, both on cold nights.",
    "Saw him over a weekend tournament.",
]
