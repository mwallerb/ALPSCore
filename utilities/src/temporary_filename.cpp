#include <alps/testing/unique_file.hpp>
#include <alps/utilities/temporary_filename.hpp>
#include <cstdlib>

namespace alps {
    std::string temporary_filename(const std::string& prefix)
    {
        std::string dir="";
        if (prefix.find('/')==prefix.npos) {
            const char* tmpdir=std::getenv("TMPDIR");
            if (tmpdir!=nullptr && tmpdir[0]!=0) {
                dir.assign(tmpdir);
                dir.append("/");
            } else {
                dir.assign("/tmp/");
            }
        }
        return alps::testing::unique_file(dir+prefix, alps::testing::unique_file::REMOVE_AND_DISOWN).name();
    }
}
