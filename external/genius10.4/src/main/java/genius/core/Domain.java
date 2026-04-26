package genius.core;

import java.util.List;
import java.util.Random;

import genius.core.issue.Issue;
import genius.core.issue.Objective;

/**
 * 
 * @author W.Pasman
 *
 */
public interface Domain {

	/**
	 * @return all objectives (note, {@link Issue} is an {@link Objective}) in
	 *         the domain.
	 */
	List<Objective> getObjectives();

	/**
	 * @return the highest level {@link Objective}.
	 */
	Objective getObjectivesRoot();

	/**
	 * 
	 * @return All {@link Issue}s in the domain, sorted to preorder.This may be
	 *         computationally expensive
	 */
	List<Issue> getIssues();

	/**
	 * @param r
	 *            random variable. if null, a new {@link Random} will be used.
	 * @return a random {@link Bid} in this domain.
	 */
	Bid getRandomBid(Random r);

	/**
	 * 
	 * @return number of all possible bids in the domain. Does not care of
	 *         constraints.
	 */
	long getNumberOfPossibleBids();

	/**
	 * @return name of this domain.
	 */
	String getName();

}
