### Chapter 1: What is Software Engineering?

software engineering is just programming with a team over time.

Software is sustainable if you are capable of reacting to whatever valuable change comes along for either technical or business reasons.

Technical debt is the delta between what our code is and what we wish it was.

refactoring and system/dependency upgrades are only important for products with a year/decade lifespan.

Rather than avoiding a painful task, invest in making it less painful. Getting through not only the first big upgrade but getting to the point at which you can reliably stay current going forward, is the essence of long-term sustainability.

Hyrum's Law: With a sufficient number of users of an API, it does not matter what you promise in the contract: all observable behaviors of your system will be depended on by somebody.

Given enough time and enough users, even the most innocous change will break something. Analysis of the value of the change must incorporate the difficulty in investigating, identifying, and resolving those breakages.

It's far too easy for problems to worsen slowly and never manifest as a singular moment of crisis (boiling frog problem).

The more frequently you change your infrastructure, the easier it becomes to do so in the future due to expertise (more people have experienced an upgrade) and stability (more frequent upgrades usually means fewer changes between releases.

Finding problems earlier in the development workflow usually reduces costs.

All decisions should have good justifications. Important to assess:

- financial costs
- resource costs
- personnel costs
- opportunity costs
- societal costs

Efficiency gains from keeping engineers happy, focused, and engaged can easily dominate other factors.

Jevons paradox: consumption of a resource may increase as a response to greater efficiency in its use. (Ex. M1 macs are an order of magnitude faster, so now we're less concerned with performance).

Every task an org has to do should be scalable. Establishing policies can make processes scalable.

### Chapter 2: How to Work Well on Teams

Software engineering is a team endeavor. Individuals need to adopt humility (be open to self-improvement), respect (treat others kindly and appreciate their abilities and accomplishments), trust (trust others are competent and will do the right thing) as core principles.

The genius myth: tendency to ascribe the success of a team to a single person/leader. Linus Torvalds, Steve Jobs, Bill Gates all have massive teams. World-changing achievements rarely occur in a vacuum.

The vast majority of development work doesn't require genius-level intellect but 100% of the work requires a minimal level of social skills.

If you spend all your time working alone, you increase the risk of unnecessary failure and cheating your potential for growth. Collaboration lets you go faster, catch fundamental design mistakes, and ensure knowledge silos don't develop.

Get feedback as early as possible, test as early as possible, and think about security and production environments as early as possible.

Many eyes makes sure your project stays relevant and on track.

Relationships outlast projects - invest and play the social game.

If you are not failing occasionally, you are not being innovative enough or taking enough risks.

Build a blameless post-mortem culture: summarize the event and its timeline from discovery to resolution, identify cause, assess damage, come up with action items for resolution, and come up with action items to prevent the event from happening again.

A good developer:
- thrives in ambiguity: can make progress towards a problem, even when the environment is constantly shifting
- values feedback: has humility to receive a give feedbkac gracefully
- puts the user first
- cares about the team - actively work to help team without being asked

### Chapter 3: Knowledge Sharing

Organizations needs a culture of learning

challenges/obstacles
- lack of psychological safety
- knowledge islands that elad to info fragmentation and duplication
- single point of failure/bus factor
- all-or-nothing expertise
- parroting - mimicry without understanding
- haunted graveyards - areas in code that people are afraid to change



You can have all the uninterrupted time in the world, but if you're using it to work on the wrong thing, you're wasting your time.

Document your knowledge so that it can scale. Documentation comes with tradeoffs - its more generalized and not tailored to individual learner situation and it has a maintenance cost.

Documentation doesn't replace human expertise - humans can assess which info is applicable to an individual's use case, determine whether the documentation is still relevant, and knows where to find it.

A big part of learning is being able to try things and feeling safe to fail.

Inheriting a legacy code base - Chesterson's fence - removing something (a fence) just because you don't see the use of it, isn't a good idea. You should only remove it if you know *why* you don't see the use of it.

Strategies:
- wikis
- group chats
- mailing lists
- stack overflow clones
- office hours
- tech talks/classes
- codelabs - guided, hands on tutorials
- static analysis tools

The first time you learn somethinhg is the best time to see ways that the existing documentation and training materials can be improved. Follow the scout rule - leave the campground (documentation) cleaner than you found it.

Bad behavior of a few individuals can make an entire team unwelcoming. Novices end up taking their questions elsewhere and new experts stop trying. In the worst cases, the group reduces to its most toxic members. Knowledge sharing should be done with kindness and respect.

A leader should improve the quality of the people around them, improve the teams psychological safety, create a culture of teamwork and collaboration.
