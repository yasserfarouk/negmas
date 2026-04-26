package negotiator.parties;

import genius.core.parties.NegotiationInfo;

public class TracerParty extends NonDeterministicConcederNegotiationParty {

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

	}

	@Override
	public double getTargetUtility() {
		double t = super.getTargetUtility();
		System.out.println(t);
		return t;
	}
}
