package genius.core.utility;

import java.io.IOException;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.xml.SimpleElement;

/**
 * This abstract class can be employed by an agent to create a custom utility space (i.e. a function
 * from a {@link Bid} in the {@link Domain} to a utility) by extending it and implementing 
 * the getUtility(Bid) method.
 */
public abstract class CustomUtilitySpace extends AbstractUtilitySpace
	{
	private static final long serialVersionUID = -9054703865527771432L;

		public CustomUtilitySpace(Domain dom) 
		{
			super(dom);
		}		

		/**
		 * All methods below are implemented as stubs, since they are not needed when 
		 * used internally by an agent
		 */
		
		@Override
		public UtilitySpace copy() {
			return null;
		}

		@Override
		public String isComplete() {
			return null;
		}

		@Override
		public SimpleElement toXML() throws IOException {
			return null;
		}
		
	}
