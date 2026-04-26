package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

/**
 * 
 * @author Dmytro Wouter: BidIterator iterates through all bids in the domain.
 *         It may result bids that do not fulfill the constraints and therefore
 *         may have utility 0.
 *
 */
public class BidIterator implements Iterator<Bid> {
	protected Domain fDomain;
	protected int fNumberOfIssues;
	protected int[] fValuesIndexes;
	protected boolean fInit;

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
			// to loop through the Real and Price Issues we use discretization
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
			/*
			 * Removed by DT because KH removed PRICE
			 * 
			 * case PRICE: IssuePrice lIssuePrice = (IssuePrice)lIssue;
			 * lNumberOfValues = lIssuePrice.getNumberOfDiscretizationSteps();
			 * break;
			 */
			}// switch
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
				/*
				 * Removed by DT because KH removed PRICE
				 * 
				 * 
				 * case PRICE: IssuePrice lIssuePrice=(IssuePrice)lIssue;
				 * lOneStep =
				 * (lIssuePrice.getUpperBound()-lIssuePrice.getLowerBound
				 * ())/lIssuePrice.getNumberOfDiscretizationSteps(); lValues[i]=
				 * new
				 * ValueReal(lIssuePrice.getLowerBound()+lOneStep*fValuesIndexes
				 * [i]); break;
				 */
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					lValues.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(fValuesIndexes[i]));
					break;
				}// switch
			}// for

			lBid = new Bid(fDomain, lValues);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return lBid;
	}

	public void remove() {
		// TODO Auto-generated method stub

	}

}
