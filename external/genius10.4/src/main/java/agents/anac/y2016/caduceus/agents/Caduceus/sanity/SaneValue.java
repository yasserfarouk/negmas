package agents.anac.y2016.caduceus.agents.Caduceus.sanity;

/**
 * AI2015Group3Assignment
 * <p/>
 * Created by Taha Doğan Güneş on 28/11/15. Copyright (c) 2015. All rights
 * reserved.
 */
public class SaneValue {
	public String name;
	public double utility;

	public SaneValue(String name, double utility) {
		this.name = name;
		this.utility = utility;
	}

	@Override
	public String toString() {
		String str = "(" + this.utility + ") ";
		return str;
	}
}
