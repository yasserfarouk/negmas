package agents.nastyagent;

import genius.core.parties.NegotiationInfo;

public class ThrowInInit extends NastyAgent {

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		throw new RuntimeException("just throwing in init for fun");
	}

}
