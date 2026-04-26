package agents.anac.y2014.E2Agent.myUtility;

import java.io.Serializable;
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
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.timeline.Timeline;
import genius.core.utility.*;

public class SessionData implements Serializable {
	private static final long serialVersionUID = 1L;

	ArrayList<BidStorageList> sessionsBitStorageList = null;
	ArrayList<Boolean> agreementList = null;
	ArrayList<BidStorage> lastBidList = null; 
	ArrayList<Double> discountedUtilList = null;
	
	public SessionData(int sessionTotal) {
		sessionsBitStorageList = new ArrayList<BidStorageList>(sessionTotal);
		agreementList = new ArrayList<Boolean>(sessionTotal);
		lastBidList = new ArrayList<BidStorage>(sessionTotal);
		discountedUtilList = new ArrayList<Double>();
	}
	
	public void save(int sessionNr, BidStorageList bidList,
			NegotiationResult result, double utility, double time) {
		sessionsBitStorageList.add(bidList);
		agreementList.add(result.isAgreement());
		System.out.println(result.isAgreement());
		lastBidList.add(new BidStorage(result.getLastBid(), utility, time));
		discountedUtilList.add(result.getMyDiscountedUtility());
	}
	
	public Parameters getParamters(double res, double dis) {
		double u = 0;
		double t = 0;
		double agreementCount = 0;
		
		int size = lastBidList.size();
		Parameters p = null;
		for(int i=0; i<size; ++i) {
			if(agreementList.get(i)) {
				agreementCount += 1;
				u += lastBidList.get(i).getUtility();
				t += lastBidList.get(i).getTime();
			}
		}
		if(agreementCount>0) {
			p = new Parameters(u/agreementCount, t/agreementCount, 
					2.5*(2/(1+dis)), 2, 0.2*(1-dis));
		} else {
			p = new Parameters(1, 0, 
					2.5*(2/(1+dis)), 2, 0.2*(1-dis));
		}
		
		p.g *= Math.pow(1.1, size-agreementCount);
		if(size > 0 && agreementList.get(size-1)==false) {
			int count = 0;
			for(int i=size-1; i>=0; --i) {
				if(agreementList.get(i)) {
					break;
				} 
				count += 1;
			}
			p.beta *= Math.pow(0.9, count);
		}

		return p;
	}
}
