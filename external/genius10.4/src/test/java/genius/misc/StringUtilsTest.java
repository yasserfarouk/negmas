package genius.misc;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import genius.core.misc.StringUtils;

@RunWith(Parameterized.class)
public class StringUtilsTest {

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays
				.asList(new Object[][] { { "Party 1", "Party 2" }, { "11", "12" }, { "-12", "-13" }, { "P1", "P2" },
						{ "P-13", "P-14" }, { "#@&*1", "#@&*2" }, { "p", "p1" }, { "1p", "1p1" }, { "1p9", "1p10" } });
	}

	private String in;
	private String out;

	public StringUtilsTest(String in, String out) {
		this.in = in;
		this.out = out;
	}

	@Test
	public void test() {
		assertEquals(out, StringUtils.increment(in));
	}

}
