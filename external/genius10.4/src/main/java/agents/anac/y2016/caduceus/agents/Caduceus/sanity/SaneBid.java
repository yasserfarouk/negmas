package agents.anac.y2016.caduceus.agents.Caduceus.sanity;

import java.util.ArrayList;
import java.util.Iterator;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.Objective;
import genius.core.issue.ValueDiscrete;

/**
 * AI2015Group3Assignment
 * <p/>
 * Created by Taha Doğan Güneş on 28/11/15. Copyright (c) 2015. All rights
 * reserved.
 */
public class SaneBid {

	ArrayList<Pair<SaneIssue, SaneValue>> pairs = new ArrayList<Pair<SaneIssue, SaneValue>>();

	public SaneBid(Bid bid, SaneUtilitySpace space) {

		for (Issue issue : bid.getIssues()) {

			Objective objective = (Objective) issue;
			String name = issue.getName();
			SaneUtilitySpace.IssueSpace issueSpace = space.saneSpaceMap
					.get(name);
			SaneIssue saneIssue = space.saneIssueMap.get(name);
			try {
				ValueDiscrete value = (ValueDiscrete) bid.getValue(objective
						.getNumber());
				SaneValue saneValue = issueSpace.findValue(value.getValue());

				Pair<SaneIssue, SaneValue> p = new Pair<SaneIssue, SaneValue>(
						saneIssue, saneValue);
				this.pairs.add(p);
			} catch (Exception e) {
				e.printStackTrace();
			}

		}

	}

	public SaneBid(ArrayList<SaneValue> values, ArrayList<SaneIssue> issues) {
		assert values.size() == issues.size();

		Iterator<SaneValue> valueIterator = values.iterator();
		Iterator<SaneIssue> issueIterator = issues.iterator();

		while (valueIterator.hasNext() && issueIterator.hasNext()) {
			SaneValue value = valueIterator.next();
			SaneIssue issue = issueIterator.next();

			Pair<SaneIssue, SaneValue> pair = new Pair<SaneIssue, SaneValue>(
					issue, value);
			pairs.add(pair);
		}

	}

	public Iterator<Pair<SaneIssue, SaneValue>> getIterator() {
		return pairs.iterator();
	}

}
