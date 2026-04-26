package agents.anac.y2019.agentlarry;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.*;

import java.util.List;

public class VectorConverter {
    /**
     * Convert a bid to vector
     *
     * First add to vector a value of one (bias)
     *
     * Then for each issue of the bid if its an integer change it to range between 0 to 1 and add it to the vector
     * if it is discrete for each possible value add a value to the vector -
     * 0 if the issue is not the value and 1 if it is the value
     * For example discrete issue of color with possible value blue, green and yellow and the real value is green
     * we will add to the vector [0,1,0]
     *
     * @param bid The bid to convert
     * @return The vector from the bid
     */
    public Vector convert(Bid bid) {
        List<Issue> issues = bid.getIssues();
        Vector vector = new Vector();
        vector.add(1.0);
        for (Issue issue : issues) {
            if (issue instanceof IssueInteger) {
                vector.add(this.convertInteger((IssueInteger) issue, bid));
            } else {
                vector.addAll(this.convertDiscrete((IssueDiscrete) issue, bid));
            }
        }

        return vector;
    }

    /**
     * @param domain The domain of the negotiation
     * @return The size of the vector
     */
    public int getVectorSize(Domain domain) {
        int vectorSize = 1;
        for (Issue issue : domain.getIssues()) {
            if (issue instanceof IssueInteger) {
                vectorSize++;
            } else {
                IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
                vectorSize += issueDiscrete.getNumberOfValues();
            }
        }
        return vectorSize;
    }

    /**
     * @param value The value to normalize
     * @param upperBound The upper bound of the value
     * @param lowerBound The lower bound of the value
     * @return The value between 0 to 1
     */
    private double normalize(int value, int upperBound, int lowerBound) {
        return (double)(value - lowerBound) / (double)(upperBound - lowerBound);
    }

    /**
     * @param issue The issue to convert
     * @param bid The bid of the issue
     * @return The converted integer
     */
    private double convertInteger(IssueInteger issue, Bid bid) {
        ValueInteger valueInteger = (ValueInteger) bid.getValue(issue.getNumber());
        return this.normalize(valueInteger.getValue(), issue.getUpperBound(),
                issue.getLowerBound());
    }

    /**
     * @param issueDiscrete The issue to convert
     * @param bid The bid of the issue
     * @return The converted discrete
     */
    private Vector convertDiscrete(IssueDiscrete issueDiscrete, Bid bid) {
        Vector vector = new Vector();
        for (ValueDiscrete valueDiscrete : issueDiscrete.getValues()) {
            if (bid.getValue(issueDiscrete.getNumber()).equals(valueDiscrete)) {
                vector.add(1.0);
            }
            else {
                vector.add(0.0);
            }
        }
        return vector;
    }
}
