#ifndef _B5_PAPI_HPP_
#define _B5_PAPI_HPP_

#include <cstdint>
#include <vector>
#include <utility>

namespace b5 {

/** extracts user-specified hardware PAPI counters. Fails gracefully
 * by returning NaN on the requested counters which are not available.
 *
 * @see PAPI wiki https://icl.cs.utk.edu/projects/papi/wiki/Main_Page
 * @see PAPI user's guide http://icl.cs.utk.edu/projects/papi/files/documentation/PAPI_USER_GUIDE_23.htm
 * @see https://www.cs.uoregon.edu/research/tau/docs/newguide/ch03s06.html
 * @see PAPI license: (end of page) http://icl.cs.utk.edu/papi/software/index.html
 *
 * (c) Joao Paulo Magalhaes 2016
 */
class PAPICounters
{
public:

    typedef std::pair< int, double > value_type;
    typedef value_type const* const_iterator;

public:

    PAPICounters();
    ~PAPICounters();

    ///
    PAPICounters(int num_events, int const* events);
    void init   (int num_events, int const* events);

    PAPICounters(std::initializer_list< int > il);
    void init   (std::initializer_list< int > il);

    void start();
    void read();
    void accum();
    void stop();

    void print(const char *prefix = nullptr);

    static const char* event_str(int evt) { return _event_str(evt); }
    static const char* event_desc(int evt) { return _event_str(evt, false); }

    const_iterator begin() const { return &m_events_asked[0]; }
    const_iterator end  () const { return &m_events_asked[0] + m_events_asked.size(); }

protected:

    std::vector< value_type > m_events_asked;
    std::vector< int        > m_events_pos;

    std::vector< int        > m_events;
    std::vector< long long  > m_results_tmp;

    uint32_t                  m_state;
    enum { _INIT = 1, _STARTED = 2 };

protected:

    void _init();
    void _extract();

    static const char* _event_str(int evt, bool just_name = true);
    static const char* _report_err(int err);

};
} // namespace b5

#endif // _B5_PAPI_HPP_
