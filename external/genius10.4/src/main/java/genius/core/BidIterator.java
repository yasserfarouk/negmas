package genius.core;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * Class used to generate all bids in the domain. For issues with a continuous
 * range discretization is used. If you want to search the set of generated bids
 * efficiently, consider using SortedOutcomeSpace instead.
 * 
 * @author Dmytro, Wouter
 */
public class BidIterator implements Iterator<Bid> {
	protected Domain fDomain;
	protected int fNumberOfIssues;
	protected int[] fValuesIndexes;
	protected boolean fInit;

	/**
	 * Creates an iterator for the given outcomespace (domain).
	 * 
	 * @param pDomain
	 *            of which we want to generate all bids.
	 */
	public BidIterator(Domain pDomain) {
		fDomain = pDomain;
		fInit = true;
		fNumberOfIssues = fDomain.getIssues().size();
		fValuesIndexes = new int[fNumberOfIssues];
	}

	public boolean hasNext() {
		int[] lNextIndexes = makeNextIndexes();
		boolean result = false;
		if (fInit) {
			return true;
		} else {
			for (int i = 0; i < fNumberOfIssues; i++)
				if (lNextIndexes[i] != 0) {
					result = true;
					break;
				}
			return result;
		}
		// return fHasNext;
	}

	private int[] makeNextIndexes() {
		int[] lNewIndexes = new int[fNumberOfIssues];
		for (int i = 0; i < fNumberOfIssues; i++)
			lNewIndexes[i] = fValuesIndexes[i];
		List<Issue> lIssues = fDomain.getIssues();
		for (int i = 0; i < fNumberOfIssues; i++) {
			Issue lIssue = lIssues.get(i);
			// to loop through the Real Issues we use discretization
			int lNumberOfValues = 0;
			switch (lIssue.getType()) {
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;
				lNumberOfValues = lIssueInteger.getUpperBound()
						- lIssueInteger.getLowerBound() + 1;
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				lNumberOfValues = lIssueReal.getNumberOfDiscretizationSteps();
				break;
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				lNumberOfValues = lIssueDiscrete.getNumberOfValues();
				break;
			}
			if (lNewIndexes[i] < lNumberOfValues - 1) {
				lNewIndexes[i]++;
				break;
			} else {
				lNewIndexes[i] = 0;
			}

		}// for
		return lNewIndexes;
	}

	public Bid next() {
		Bid lBid = null;
		int[] lNextIndexes = makeNextIndexes();
		if (fInit)
			fInit = false;
		else
			fValuesIndexes = lNextIndexes;

		// build Hashmap and createFrom the next bid.
		try {
			HashMap<Integer, Value> lValues = new HashMap<Integer, Value>(/*
																		 * 16,(float
																		 * )0.75
																		 */);
			List<Issue> lIssues = fDomain.getIssues();
			for (int i = 0; i < fNumberOfIssues; i++) {
				Issue lIssue = lIssues.get(i);
				double lOneStep;
				switch (lIssue.getType()) {
				// TODO: COMPLETE add cases for all types of issues
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					lValues.put(lIssue.getNumber(), new ValueInteger(
							lIssueInteger.getLowerBound() + fValuesIndexes[i]));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					lOneStep = (lIssueReal.getUpperBound() - lIssueReal
							.getLowerBound())
							/ (lIssueReal.getNumberOfDiscretizationSteps() - 1);
					lValues.put(lIssue.getNumber(),
							new ValueReal(lIssueReal.getLowerBound() + lOneStep
									* fValuesIndexes[i]));
					break;
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					lValues.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(fValuesIndexes[i]));
					break;
				}
			}

			lBid = new Bid(fDomain, lValues);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return lBid;
	}

	public void remove() {
	}

}
