package  agents.anac.y2014.E2Agent.myUtility;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import agents.anac.y2011.TheNegotiator.BidsCollection;
import agents.anac.y2012.MetaAgent.agents.WinnerAgent.opponentOffers;
import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.timeline.Timeline;
import genius.core.utility.*;





public class SummaryStatistics {
    private double average = 0;
    private double variance = 0;

    public SummaryStatistics(double ave, double var) {
        average = ave;
        variance = var;
    }

    public double getAve() { return average; }
    public double getVar() { return variance; }
    public String toString() { return "Ave: " + average + " Var: " + variance; }
}
