package agents.anac.y2016.caduceus.agents.Caduceus.sanity;

/**
 * AI2015Group3Assignment
 * <p/>
 * Created by Taha Doğan Güneş on 28/11/15. Copyright (c) 2015. All rights
 * reserved.
 */
public class SaneIssue { // Season
	public double weight;
	public String name;

	public SaneIssue(String name) { // if this is from bid
		this.name = name;
	}

	public SaneIssue(double weight, String name) {
		this.weight = weight;
		this.name = name;
	}

}
