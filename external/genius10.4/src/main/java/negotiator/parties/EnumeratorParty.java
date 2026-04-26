package negotiator.parties;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.logging.CsvLogger;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * Debug class for enumerating all offers to a file
 */
public class EnumeratorParty extends AbstractNegotiationParty {

	public static int id = 0;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		// vars
		Domain dom = getUtilitySpace().getDomain();
		List<Integer> issueSizes = new ArrayList<Integer>();
		for (Issue issue : dom.getIssues())
			issueSizes.add(((IssueDiscrete) issue).getNumberOfValues());

		System.out.printf("Enumerating all %d possible offers\n", dom.getNumberOfPossibleBids());
		System.out.printf("There are %d issues with the following number of values: %s\n", dom.getIssues().size(),
				issueSizes.toString());
		try {
			String party = "unknown";
			switch (id) {
			case 0:
				party = "Minister";
				break;
			case 1:
				party = "NS";
				break;
			case 2:
				party = "ProRail";
				break;
			}

			id++;
			CsvLogger logger = new CsvLogger(String.format("Railway-%s-utils.csv", party));

			for (int issueIndex = 0; issueIndex < issueSizes.size(); issueIndex++) {
				logger.log(String.format("\"Issue %d\"", issueIndex + 1));
			}
			logger.logLine("utility");

			for (int bidIndex = 0; bidIndex < dom.getNumberOfPossibleBids(); bidIndex++) {
				Integer[] issueIndices = new Integer[issueSizes.size()];
				Arrays.fill(issueIndices, 0);
				int remainder = bidIndex;
				int currentIndex = 0;

				while (remainder > 0) {
					int size = issueSizes.get(currentIndex);
					int div = remainder / size;
					int mod = remainder % size;

					issueIndices[currentIndex] = mod;
					remainder = div;
					currentIndex++;
				}

				Bid bid = generateBid(issueIndices);
				double util = getUtility(bid);

				System.out.printf("%s -> %.3f\n", new ArrayList<Integer>(Arrays.asList(issueIndices)), util);
				for (Integer issueIndex : issueIndices) {
					logger.log(issueIndex);
				}
				logger.logLine(util);
			}

			logger.close();
		} catch (FileNotFoundException e) {
			System.err.println("Problems starting EnumeratorParty logger");
			e.printStackTrace();
		} catch (IOException e) {
			System.err.println("Problems closing EnumeratorParty logger");
			e.printStackTrace();
		}

		System.out.println("Enumeration finished\n");
	}

	public Bid generateBid(Integer[] indices) {

		HashMap<Integer, Value> values = new HashMap<Integer, Value>();

		for (int i = 0; i < indices.length; i++) {
			IssueDiscrete issue = (IssueDiscrete) getUtilitySpace().getDomain().getIssues().get(i);
			values.put(i + 1, issue.getValue(indices[i]));
		}

		try {
			return new Bid(getUtilitySpace().getDomain(), values);
		} catch (Exception e) {
			System.err.println("Could not generate offer");
			return null;
		}
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		return new Offer(getPartyId(), generateRandomBid());
	}

	@Override
	public void receiveMessage(AgentID sender, Action arguments) {
		// do nothing
	}

	@Override
	public String getDescription() {
		return "Enumerator Party for Debug";
	}
}
