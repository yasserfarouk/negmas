package parties.simplemediator;

import java.util.Random;

import genius.core.parties.NegotiationInfo;

/**
 * This mediator behaves exactly like {@link FixedOrderFlippingMediator} except
 * that it uses for its random seed Random(0), so its "random" moves will always
 * be the same..
 */
public class FixedOrderFlippingMediator extends RandomFlippingMediator {

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		rand = new Random(0);
	}

}
